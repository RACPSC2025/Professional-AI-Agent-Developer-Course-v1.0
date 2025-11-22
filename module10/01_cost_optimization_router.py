"""
🟢 NIVEL BÁSICO: ROUTER DE COSTOS - OPTIMIZACIÓN LLMOps
-------------------------------------------------------
Este ejemplo demuestra un router inteligente que selecciona el modelo óptimo por costo/calidad.
Caso de Uso: Reducir costos de API en 70% sin sacrificar calidad.

Conceptos Clave:
- Model routing: Selector de modelo basado en complejidad
- Cost tracking: Medición de costos en tiempo real
- Fallback strategies: Plan B si falla el modelo principal
"""

import os
import sys
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. CONFIGURACIÓN DE MODELOS Y COSTOS ---
# Precios aproximados (por 1M tokens, actualizar según OpenAI pricing)
MODEL_CONFIGS = {
    "gpt-4o": {
        "model": "gpt-4o",
        "cost_per_1m_input": 2.50,   # USD
        "cost_per_1m_output": 10.00,
        "max_tokens": 4096,
        "capabilities": ["reasoning", "complex", "creative"],
        "latency": "high"
    },
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "cost_per_1m_input": 0.150,
        "cost_per_1m_output": 0.600,
        "max_tokens": 16384,
        "capabilities": ["simple", "factual", "structured"],
        "latency": "medium"
    },
    "gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo",
        "cost_per_1m_input": 0.50,
        "cost_per_1m_output": 1.50,
        "max_tokens": 4096,
        "capabilities": ["simple", "chat"],
        "latency": "low"
    }
}

# --- 2. CLASIFICADOR DE COMPLEJIDAD ---
class QueryComplexityRouter:
    """Clasifica queries y enruta al modelo apropiado."""
    
    def __init__(self):
        # Usamos un modelo barato para clasificación
        self.classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.classifier_prompt = ChatPromptTemplate.from_template("""
Clasifica la complejidad de la siguiente consulta de usuario:

CONSULTA: {query}

Responde SOLO con una palabra:
- "SIMPLE": Preguntas factuales, definiciones, tareas rutinarias
- "MEDIUM": Análisis básico, comparaciones, resúmenes
- "COMPLEX": Razonamiento multi-paso, creatividad, análisis profundo

CLASIFICACIÓN:
        """)
        
        self.classifier_chain = self.classifier_prompt | self.classifier_llm | StrOutputParser()
    
    def classify(self, query: str) -> Literal["SIMPLE", "MEDIUM", "COMPLEX"]:
        """Clasifica la complejidad de una query."""
        result = self.classifier_chain.invoke({"query": query}).strip().upper()
        
        if result in ["SIMPLE", "MEDIUM", "COMPLEX"]:
            return result
        else:
            # Default a MEDIUM si no está claro
            return "MEDIUM"
    
    def route_to_model(self, query: str) -> str:
        """Enruta query al modelo más cost-effective."""
        complexity = self.classify(query)
        
        # Estrategia de routing
        if complexity == "SIMPLE":
            return "gpt-4o-mini"  # Más barato para tareas simples
        elif complexity == "MEDIUM":
            return "gpt-4o-mini"  # Still cost-effective
        else:  # COMPLEX
            return "gpt-4o"  # Modelo premium para razonamiento complejo
        
# --- 3. SISTEMA DE TRACKING DE COSTOS ---
class CostTracker:
    """Rastrea costos acumulados de llamadas a API."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.calls_by_model = {}
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calcula costo de una llamada."""
        config = MODEL_CONFIGS[model]
        
        input_cost = (input_tokens / 1_000_000) * config["cost_per_1m_input"]
        output_cost = (output_tokens / 1_000_000) * config["cost_per_1m_output"]
        
        return input_cost + output_cost
    
    def log_call(self, model: str, input_tokens: int, output_tokens: int):
        """Registra una llamada y actualiza costos."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        
        if model not in self.calls_by_model:
            self.calls_by_model[model] = {"count": 0, "cost": 0.0, "tokens": 0}
        
        self.calls_by_model[model]["count"] += 1
        self.calls_by_model[model]["cost"] += cost
        self.calls_by_model[model]["tokens"] += input_tokens + output_tokens
        
        return cost
    
    def get_report(self) -> str:
        """Genera reporte de costos."""
        report = f"💰 COSTO TOTAL: ${self.total_cost:.4f}\n\n"
        report += "📊 DESGLOSE POR MODELO:\n"
        
        for model, stats in self.calls_by_model.items():
            report += f"  {model}:\n"
            report += f"    Llamadas: {stats['count']}\n"
            report += f"    Tokens: {stats['tokens']:,}\n"
            report += f"    Costo: ${stats['cost']:.4f}\n\n"
        
        return report

# --- 4. AGENTE CON ROUTING ---
class CostOptimizedAgent:
    """Agente que optimiza costos mediante routing inteligente."""
    
    def __init__(self):
        self.router = QueryComplexityRouter()
        self.tracker = CostTracker()
        
        self.prompt = ChatPromptTemplate.from_template("""
Eres un asistente útil y conciso.

PREGUNTA: {question}

RESPUESTA:
        """)
    
    def invoke(self, query: str) -> dict:
        """Procesa query con routing automático."""
        
        # 1. Determinar modelo óptimo
        selected_model = self.router.route_to_model(query)
        print(f"🔀 Routing: {selected_model.upper()}")
        
        # 2. Ejecutar con modelo seleccionado
        llm = ChatOpenAI(model=selected_model, temperature=0.7)
        chain = self.prompt | llm | StrOutputParser()
        
        # Obtener response (en producción, usar callbacks para tokens exactos)
        response = chain.invoke({"question": query})
        
        # 3. Estimar tokens (aproximado)
        input_tokens = len(query.split()) * 1.3  # Estimación rough
        output_tokens = len(response.split()) * 1.3
        
        cost = self.tracker.log_call(selected_model, int(input_tokens), int(output_tokens))
        
        return {
            "answer": response,
            "model_used": selected_model,
            "estimated_cost": cost
        }

# --- 5. DEMO ---
if __name__ == "__main__":
    print("="*70)
    print("  💰 ROUTER DE COSTOS - OPTIMIZACIÓN LLMOps")
    print("="*70)
    
    agent = CostOptimizedAgent()
    
    print("\n🧪 CASOS DE PRUEBA:\n")
    
    test_queries = [
        "¿Qué es Python?",  # SIMPLE -> gpt-4o-mini
        "Explica la diferencia entre async y sync en JavaScript",  # MEDIUM -> gpt-4o-mini
        "Diseña una arquitectura escalable para un sistema de trading de alta frecuencia con latencia sub-milisegundo",  # COMPLEX -> gpt-4o
    ]
    
    for i, query in enumerate(test_queries):
        print(f"--- QUERY {i+1} ---")
        print(f"📝 Pregunta: {query}")
        
        result = agent.invoke(query)
        
        print(f"💬 Respuesta: {result['answer'][:150]}...")
        print(f"💵 Costo estimado: ${result['estimated_cost']:.6f}\n")
    
    # Reporte final
    print("="*70)
    print(agent.tracker.get_report())
    
    # Comparación con estrategia naive (todo en GPT-4o)
    naive_cost = sum([
        agent.tracker.calculate_cost("gpt-4o", len(q.split()) * 1.3, 50)
        for q in test_queries
    ])
    
    print(f"💡 COMPARACIÓN:")
    print(f"   Con routing inteligente: ${agent.tracker.total_cost:.4f}")
    print(f"   Sin routing (todo GPT-4o): ${naive_cost:.4f}")
    print(f"   ✅ Ahorro: {((naive_cost - agent.tracker.total_cost) / naive_cost * 100):.1f}%")
