"""
01_instrumentation_basics.py
============================
Instrumentación Básica con LangSmith.

Este script demuestra cómo "encender la luz" en tu agente.
Al ejecutarlo, cada paso (llamada a LLM, uso de herramienta) se enviará a LangSmith.

Requisitos:
1. Crear cuenta en https://smith.langchain.com/
2. Obtener API Key.
3. pip install langchain langchain-openai langsmith

Variables de Entorno Requeridas:
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
- LANGCHAIN_API_KEY="<tu-api-key>"
- LANGCHAIN_PROJECT="curso-agentes-module11"
"""

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LangChainTracer

# Configuración (Idealmente cargar desde .env)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Module 11 Demo"
# os.environ["LANGCHAIN_API_KEY"] = "sk-..." # Asegúrate de tener esto setead

def build_agent():
    """Construye una cadena simple para demostración."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en observabilidad de software. Sé técnico pero claro."),
        ("user", "{question}")
    ])
    
    # Chain simple: Prompt -> LLM -> String
    chain = prompt | llm | StrOutputParser()
    return chain

if __name__ == "__main__":
    print("🕵️  Iniciando Trace Demo con LangSmith...")
    
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("⚠️  ADVERTENCIA: No se detectó LANGCHAIN_API_KEY. El tracing fallará o no se verá.")
    
    agent = build_agent()
    
    questions = [
        "¿Por qué es importante el distributed tracing en microservicios?",
        "Explica la diferencia entre métricas y logs."
    ]
    
    for q in questions:
        print(f"\n❓ Preguntando: {q}")
        start = time.time()
        
        # Al invocar la cadena, LangChain envía automáticamente los datos a LangSmith
        # gracias a las variables de entorno.
        response = agent.invoke({"question": q})
        
        end = time.time()
        print(f"✅ Respuesta ({end-start:.2f}s): {response[:100]}...")
    
    print("\n✨ Ve a https://smith.langchain.com/ para ver tus trazas en tiempo real.")
