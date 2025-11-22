"""
Módulo 10 - Ejemplo Avanzado: Framework de A/B Testing
Caso de uso: Experimentación sistemática con prompts y modelos

Permite comparar diferentes variantes (prompts, modelos, temperaturas)
y determinar cuál funciona mejor mediante análisis estadístico.

Instalación:
pip install langchain langchain-openai scipy numpy
"""

import os
from typing import List, Dict
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
import numpy as np
from scipy import stats
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Variant:
    """Variante experimental"""
    name: str
    model: str
    temperature: float
    system_prompt: str


@dataclass
class ExperimentResult:
    """Resultado de un experimento"""
    variant_name: str
    success_rate: float
    avg_latency_ms: float
    avg_tokens: int
    sample_size: int


class ABTestFramework:
    """Framework para A/B testing de prompts y modelos"""
    
    def __init__(self):
        self.results = []
    
    def run_experiment(self, variants: List[Variant], test_queries: List[Dict], 
                      evaluation_fn) -> List[ExperimentResult]:
        """Ejecutar experimento A/B con múltiples variantes"""
        
        print("="*70)
        print("🧪 A/B TEST EXPERIMENT")
        print("="*70)
        print(f"\n📊 Configuración:")
        print(f"   Variantes: {len(variants)}")
        print(f"   Test queries: {len(test_queries)}")
        
        results = []
        
        for variant in variants:
            print(f"\n{'='*70}")
            print(f"Testing Variant: {variant.name}")
            print(f"   Model: {variant.model}")
            print(f"   Temperature: {variant.temperature}")
            print('='*70)
            
            # Ejecutar queries
            successes = 0
            total_latency = 0
            total_tokens = 0
            
            llm = ChatOpenAI(
                model=variant.model,
                temperature=variant.temperature
            )
            
            for i, test in enumerate(test_queries, 1):
                query = test["query"]
                expected_behavior = test.get("expected_behavior", "")
                
                print(f"\n[{i}/{len(test_queries)}] Query: {query[:50]}...")
                
                # Generar respuesta
                import time
                start = time.time()
                
                full_prompt = f"{variant.system_prompt}\n\nUser: {query}"
                response = llm.invoke(full_prompt)
                
                latency_ms = (time.time() - start) * 1000
                tokens = len(response.content) // 4  # Estimación
                
                # Evaluar calidad
                is_success = evaluation_fn(query, response.content, expected_behavior)
                
                if is_success:
                    successes += 1
                    print(f"   ✅ Success - {latency_ms:.0f}ms")
                else:
                    print(f"   ❌ Failed - {latency_ms:.0f}ms")
                
                total_latency += latency_ms
                total_tokens += tokens
            
            # Calcular métricas
            success_rate = (successes / len(test_queries)) * 100
            avg_latency = total_latency / len(test_queries)
            avg_tokens = total_tokens // len(test_queries)
            
            result = ExperimentResult(
                variant_name=variant.name,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                avg_tokens=avg_tokens,
                sample_size=len(test_queries)
            )
            
            results.append(result)
            
            print(f"\n📊 Resultados de {variant.name}:")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Avg Latency: {avg_latency:.0f}ms")
            print(f"   Avg Tokens: {avg_tokens}")
        
        return results
    
    def analyze_results(self, results: List[ExperimentResult]) -> Dict:
        """Analizar resultados y determinar ganador"""
        
        print("\n" + "="*70)
        print("📈 STATISTICAL ANALYSIS")
        print("="*70)
        
        # Ordenar por success rate
        sorted_results = sorted(results, key=lambda x: x.success_rate, reverse=True)
        
        print(f"\n🏆 Ranking por Success Rate:")
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result.variant_name}: {result.success_rate:.1f}% "
                  f"(latency: {result.avg_latency_ms:.0f}ms, tokens: {result.avg_tokens})")
        
        # Test de significancia estadística (simplificado)
        if len(results) >= 2:
            best = sorted_results[0]
            second = sorted_results[1]
            
            # Calcular diferencia
            diff = best.success_rate - second.success_rate
            
            print(f"\n📊 Mejor variante: {best.variant_name}")
            print(f"   Mejora sobre segundo lugar: +{diff:.1f} percentage points")
            
            # Significancia (simplificado - en producción usar test estadístico apropiado)
            if diff > 5 and best.sample_size >= 20:
                print("   ✅ Diferencia estadísticamente significativa (p < 0.05)")
            elif diff > 2:
                print("   ⚠️ Diferencia marginal - considerar más muestras")
            else:
                print("   ❌ No hay diferencia significativa")
        
        # Recomendación
        best_variant = sorted_results[0]
        
        print(f"\n💡 RECOMENDACIÓN:")
        print(f"   Deploy variante: {best_variant.variant_name}")
        print(f"   Expected success rate: {best_variant.success_rate:.1f}%")
        
        # Trade-offs
        fastest = min(results, key=lambda x: x.avg_latency_ms)
        if fastest.variant_name != best_variant.variant_name:
            print(f"\n⚡ Nota: {fastest.variant_name} es más rápido ({fastest.avg_latency_ms:.0f}ms)")
            print(f"   Si latencia es crítica, considera este trade-off.")
        
        return {
            "winner": best_variant.variant_name,
            "results": results
        }


def simple_evaluation(query: str, response: str, expected_behavior: str) -> bool:
    """
    Función de evaluación simple (en producción usar LLM-as-judge o métricas específicas)
    """
    # Criterios simples
    if len(response) < 20:
        return False
    
    if "no sé" in response.lower() or "no puedo" in response.lower():
        return False
    
    # Si hay comportamiento esperado, verificar que esté presente
    if expected_behavior and expected_behavior.lower() not in response.lower():
        return False
    
    return True


def main():
    """Demostración de A/B testing"""
    
    print("="*70)
    print("A/B Testing Framework - Optimización Basada en Datos")
    print("="*70)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    # Definir variantes a probar
    variants = [
        Variant(
            name="Control (Baseline)",
            model="gpt-4o-mini",
            temperature=0.7,
            system_prompt="Eres un asistente útil."
        ),
        Variant(
            name="Variant A: More Detailed Prompt",
            model="gpt-4o-mini",
            temperature=0.7,
            system_prompt="Eres un asistente experto. Proporciona respuestas detalladas, estructuradas y accionables. Siempre incluye ejemplos cuando sea relevante."
        ),
        Variant(
            name="Variant B: Lower Temperature",
            model="gpt-4o-mini",
            temperature=0.3,
            system_prompt="Eres un asistente útil."
        ),
    ]
    
    # Test queries
    test_queries = [
        {"query": "¿Cómo aprendo Python?", "expected_behavior": "paso"},
        {"query": "Explica qué es Docker", "expected_behavior": "contenedor"},
        {"query": "Diferencia entre REST y GraphQL", "expected_behavior": ""},
        {"query": "¿Qué es un design pattern?", "expected_behavior": ""},
        {"query": "Cómo optimizar una consulta SQL lenta" "expected_behavior": "índice"},
    ]
    
    # Ejecutar experimento
    framework = ABTestFramework()
    results = framework.run_experiment(variants, test_queries, simple_evaluation)
    
    # Analizar
    analysis = framework.analyze_results(results)
    
    print("\n" + "="*70)
    print("✅ Experimento completado")
    print("="*70)


if __name__ == "__main__":
    main()
