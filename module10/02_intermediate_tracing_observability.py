"""
Módulo 10 - Ejemplo Intermedio: Sistema de Tracing y Observabilidad
Framework: LangSmith + LangChain
Caso de uso: Dashboard de monitoreo en tiempo real

Este sistema demuestra cómo implementar observabilidad completa con métricas
de latencia, costo, y calidad de respuestas.

Instalación:
pip install langchain langchain-openai langsmith
"""

import os
import time
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Configurar LangSmith (opcional pero recomendado para producción)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-monitoring"
# Asegúrate de tener LANGCHAIN_API_KEY configurada


class ProductionMonitor:
    """Monitor de métricas de producción"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "errors": 0,
            "requests_by_model": {},
        }
    
    def record_request(self, model: str, tokens: int, latency_ms: float, cost_usd: float, error: bool = False):
        """Registrar una petición"""
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_cost_usd"] += cost_usd
        self.metrics["total_latency_ms"] += latency_ms
        
        if error:
            self.metrics["errors"] += 1
        
        if model not in self.metrics["requests_by_model"]:
            self.metrics["requests_by_model"][model] = 0
        self.metrics["requests_by_model"][model] += 1
    
    def get_summary(self) -> Dict:
        """Obtener resumen de métricas"""
        avg_latency = (self.metrics["total_latency_ms"] / self.metrics["total_requests"] 
                      if self.metrics["total_requests"] > 0 else 0)
        
        avg_tokens = (self.metrics["total_tokens"] / self.metrics["total_requests"]
                     if self.metrics["total_requests"] > 0 else 0)
        
        error_rate = (self.metrics["errors"] / self.metrics["total_requests"] * 100
                     if self.metrics["total_requests"] > 0 else 0)
        
        return {
            **self.metrics,
            "avg_latency_ms": avg_latency,
            "avg_tokens_per_request": avg_tokens,
            "error_rate_percent": error_rate
        }
    
    def print_dashboard(self):
        """Imprimir dashboard de métricas"""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("📊 PRODUCTION METRICS DASHBOARD")
        print("=" * 70)
        print(f"\n📈 Request Statistics:")
        print(f"   Total Requests: {summary['total_requests']}")
        print(f"   Errors: {summary['errors']} ({summary['error_rate_percent']:.2f}% error rate)")
        
        print(f"\n⏱️ Performance:")
        print(f"   Avg Latency: {summary['avg_latency_ms']:.0f} ms")
        print(f"   Total Latency: {summary['total_latency_ms']/1000:.2f} seconds")
        
        print(f"\n💰 Cost Metrics:")
        print(f"   Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Avg Tokens/Request: {summary['avg_tokens_per_request']:.0f}")
        print(f"   Cost per 1K tokens: ${summary['total_cost_usd']/(summary['total_tokens']/1000):.4f}" 
              if summary['total_tokens'] > 0 else "")
        
        print(f"\n🤖 Model Distribution:")
        for model, count in summary['requests_by_model'].items():
            pct = (count / summary['total_requests'] * 100) if summary['total_requests'] > 0 else 0
            print(f"   {model}: {count} requests ({pct:.1f}%)")
        
        print("=" * 70 + "\n")


def estimate_cost(model: str, tokens: int) -> float:
    """Estimar costo basado en modelo y tokens (precios aproximados)"""
    pricing = {
        "gpt-4o": {"input": 0.0025, "output": 0.010},  # per 1K tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
    # Simplificación: asumimos 50/50 input/output
    avg_price = (model_pricing["input"] + model_pricing["output"]) / 2
    return (tokens / 1000) * avg_price


def create_traced_chain(model: str):
    """Crear chain con tracing automático"""
    llm = ChatOpenAI(model=model, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil y conciso."),
        ("user", "{question}")
    ])
    
    chain = {"question": RunnablePassthrough()} | prompt | llm
    
    return chain


def main():
    """Demostración de sistema de observabilidad"""
    print("=" * 70)
    print("Sistema de Tracing y Observabilidad para Producción")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    # Crear monitor
    monitor = ProductionMonitor()
    
    # Simular diferentes tipos de requests
    test_scenarios = [
        {"model": "gpt-4o-mini", "question": "¿Qué es Python?"},
        {"model": "gpt-4o-mini", "question": "Dame 3 beneficios de usar Docker"},
        {"model": "gpt-4o", "question": "Explica en detalle cómo funcionan los transformers en NLP"},
        {"model": "gpt-4o-mini", "question": "¿Cuál es la capital de Francia?"},
        {"model": "gpt-4o-mini", "question": "Lista 5 lenguajes de programación"},
        {"model": "gpt-4o", "question": "Diseña una arquitectura de microservicios para e-commerce"},
    ]
    
    print(f"\n🚀 Ejecutando {len(test_scenarios)} requests simulados...\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        model = scenario["model"]
        question = scenario["question"]
        
        print(f"[{i}/{len(test_scenarios)}] {model}: {question[:50]}...")
        
        # Crear chain
        chain = create_traced_chain(model)
        
        # Medir latencia
        start_time = time.time()
        
        try:
            # Ejecutar (esto automáticamente se registra en LangSmith si está configurado)
            result = chain.invoke(question)
            response = result.content
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimar tokens (simplificado: contar caracteres / 4)
            estimated_tokens = (len(question) + len(response)) // 4
            
            # Estimar costo
            cost = estimate_cost(model, estimated_tokens)
            
            # Registrar métricas
            monitor.record_request(model, estimated_tokens, latency_ms, cost)
            
            print(f"   ✅ Success - {latency_ms:.0f}ms, ~{estimated_tokens} tokens, ${cost:.5f}")
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            monitor.record_request(model, 0, latency_ms, 0, error=True)
            print(f"   ❌ Error: {str(e)}")
    
    # Mostrar dashboard
    monitor.print_dashboard()
    
    # Insights
    print("💡 INSIGHTS:")
    summary = monitor.get_summary()
    
    if summary['error_rate_percent'] > 5:
        print("   ⚠️ Alta tasa de errores detectada - revisar logs")
    
    if summary['avg_latency_ms'] > 3000:
        print("   ⚠️ Latencia promedio alta - considerar caching o modelo más rápido")
    
    # Calcular savings potenciales
    gpt4_requests = summary['requests_by_model'].get('gpt-4o', 0)
    if gpt4_requests > 0:
        potential_savings = gpt4_requests * 0.01  # Estimación rough
        print(f"   💰 Potencial ahorro si se usa GPT-4o-mini: ~${potential_savings:.3f}")
    
    print("\n📝 Nota: Todas las llamadas se registran en LangSmith si está configurado.")
    print("   Visita https://smith.langchain.com para ver traces detallados.")


if __name__ == "__main__":
    main()
