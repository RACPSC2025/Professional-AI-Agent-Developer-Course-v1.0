"""
🔴 NIVEL AVANZADO: EL ENRUTADOR INTELIGENTE (THE SMART ROUTER)
--------------------------------------------------------------
No uses un cañón para matar una mosca.
Este script implementa un patrón de arquitectura "Router".
1. Un modelo pequeño y rápido (Router) analiza la intención.
2. Redirige la petición al "Worker" adecuado (Simple vs Complejo).

Conceptos:
- Classification: Determinar la categoría del problema.
- Routing: Despachar lógica condicional.
- Cost Optimization: Ahorrar dinero usando modelos baratos para tareas fáciles.
"""

import os
from enum import Enum
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# --- 1. DEFINICIÓN DE RUTAS ---
class RouteType(str, Enum):
    SIMPLE_CHAT = "simple_chat"
    DEEP_RESEARCH = "deep_research"
    CODE_GENERATION = "code_generation"

class RouterDecision(BaseModel):
    """Modelo de decisión que debe tomar el Router."""
    route: RouteType = Field(description="La ruta adecuada para la consulta")
    reasoning: str = Field(description="Breve explicación de por qué eligió esta ruta")
    complexity_score: int = Field(description="Puntuación de 1-10 de la complejidad estimada")

# --- 2. EL AGENTE ROUTER ---
class SmartRouter:
    def __init__(self):
        # Usamos un modelo rápido y barato para el routing
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=RouterDecision)
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            Eres un despachador inteligente de IA. Tu trabajo es clasificar la consulta del usuario
            y asignarla al departamento correcto.
            
            RUTAS DISPONIBLES:
            - simple_chat: Saludos, chistes, preguntas de conocimiento general básico.
            - deep_research: Preguntas que requieren buscar en internet, análisis financiero, noticias recientes.
            - code_generation: Peticiones de escribir scripts, funciones, SQL o depuración.
            
            CONSULTA DEL USUARIO:
            {query}
            
            {format_instructions}
            """
        )
        
        self.chain = self.prompt | self.llm | self.parser

    def route_query(self, query: str) -> RouterDecision:
        print(f"🚦 Analizando tráfico para: '{query}'...")
        return self.chain.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })

# --- 3. LOS WORKERS (Simulados) ---
def handle_simple_chat(query):
    print(f"   🟢 [Worker: Simple] Respondiendo rápido con GPT-4o-mini a: {query}")
    # Aquí iría una llamada real a llm.invoke(query)
    return "Hola! Soy el bot simple. Todo bien."

def handle_deep_research(query):
    print(f"   🟡 [Worker: Research] Iniciando CrewAI para investigar: {query}")
    print("      -> Creando Agente Investigador...")
    print("      -> Buscando en DuckDuckGo...")
    # Aquí iría la inicialización de una Crew
    return "Informe de investigación completo generado."

def handle_code_generation(query):
    print(f"   🔴 [Worker: Coder] Iniciando AutoGen para programar: {query}")
    print("      -> Ejecutando contenedor Docker...")
    # Aquí iría un AutoGen UserProxy
    return "Código generado y testeado en sandbox."

# --- 4. ORQUESTADOR PRINCIPAL ---
def main_system(query: str):
    router = SmartRouter()
    
    try:
        decision = router.route_query(query)
        
        print(f"   📋 Decisión: {decision.route.value.upper()} (Complejidad: {decision.complexity_score}/10)")
        print(f"   💭 Razón: {decision.reasoning}")
        
        # Switch Case (Python 3.10+)
        if decision.route == RouteType.SIMPLE_CHAT:
            handle_simple_chat(query)
        elif decision.route == RouteType.DEEP_RESEARCH:
            handle_deep_research(query)
        elif decision.route == RouteType.CODE_GENERATION:
            handle_code_generation(query)
            
    except Exception as e:
        print(f"❌ Error en routing: {e}")

if __name__ == "__main__":
    print("--- 🔀 SMART ROUTER SYSTEM ---")
    
    queries = [
        "Cuéntame un chiste corto",
        "Escribe un script en Python para scrapear Amazon",
        "Investiga el impacto del precio del litio en las acciones de Tesla en 2024"
    ]
    
    for q in queries:
        print("-" * 50)
        main_system(q)
