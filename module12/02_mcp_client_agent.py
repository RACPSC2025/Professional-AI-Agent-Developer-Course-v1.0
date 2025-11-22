"""
02_mcp_client_agent.py
======================
Cliente MCP con LangChain.

Este script demuestra cómo un Agente puede conectarse a un Servidor MCP,
descubrir sus herramientas dinámicamente y usarlas.

NOTA: Para que esto funcione, el servidor (01_mcp_server_simple.py) debe estar corriendo.
Aquí simularemos la conexión directa para propósitos educativos, ya que MCP
generalmente corre sobre stdio entre procesos.

Requisitos:
pip install langchain langchain-openai
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool

# --- Simulación de Herramientas MCP ---
# En un escenario real, estas herramientas se cargarían dinámicamente del servidor MCP.
# Aquí las envolvemos manualmente para demostrar el consumo por parte del agente.

@tool
def list_products_tool() -> str:
    """Lista todos los productos disponibles en el inventario."""
    # Simula llamada al servidor MCP
    return "- laptop_pro: $1200 (Stock: 50)\n- mouse_gamer: $50 (Stock: 100)\n- monitor_4k: $400 (Stock: 20)"

@tool
def get_product_details_tool(product_name: str) -> str:
    """Obtiene detalles de un producto."""
    # Simula llamada al servidor MCP
    if product_name == "laptop_pro":
        return "Precio: $1200, Stock: 50"
    return "Producto no encontrado"

# --- Configuración del Agente ---

def run_mcp_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # El agente "ve" las herramientas del servidor MCP
    tools = [list_products_tool, get_product_details_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de inventario. Usa las herramientas disponibles."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("🤖 Agente Conectado al Servidor MCP (Simulado)")
    
    # Caso de uso: El usuario pide algo que requiere consultar el inventario
    query = "¿Cuánto cuesta la laptop pro y cuántas quedan?"
    print(f"\nUsuario: {query}")
    
    agent_executor.invoke({"input": query})

if __name__ == "__main__":
    run_mcp_agent()
