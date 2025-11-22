"""
02_local_observability_phoenix.py
=================================
Observabilidad Local con Arize Phoenix.

Este script levanta un servidor local de observabilidad (OpenTelemetry compatible).
Ideal para entornos donde no se pueden enviar datos a la nube (Bancos, Salud, etc.).

Requisitos:
pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-sdk
"""

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# --- 1. Iniciar Phoenix (El servidor de observabilidad) ---
session = px.launch_app()
print(f"🚀 Phoenix UI está corriendo en: {session.url} (http://localhost:6006)")

# --- 2. Instrumentación Automática ---
# Esto conecta LangChain con Phoenix automáticamente
LangChainInstrumentor().instrument()

# --- 3. Definir un Agente con Herramientas (Para ver trazas complejas) ---

@tool
def multiply(a: int, b: int) -> int:
    """Multiplica dos números."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Obtiene el clima simulado."""
    return f"Soleado en {city}, 25°C"

def run_agent_workload():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [multiply, get_weather]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Construcción manual del agente (estilo moderno)
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    
    from langchain.agents import AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Ejecutar consultas para generar trazas
    queries = [
        "¿Cuánto es 55 * 4?",
        "¿Qué tiempo hace en Madrid y cuánto es 10 * 10?"
    ]
    
    for q in queries:
        print(f"\n🤖 Ejecutando: {q}")
        agent_executor.invoke({"input": q})

if __name__ == "__main__":
    try:
        run_agent_workload()
        print("\n✅ Trazas enviadas.")
        print("🛑 Presiona Ctrl+C para detener el servidor Phoenix.")
        input("Presiona Enter para salir...")
    except Exception as e:
        print(f"Error: {e}")
