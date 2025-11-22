"""
🟡 NIVEL INTERMEDIO: ANALISTA DE INVERSIONES (THE AGENT)
--------------------------------------------------------
Este script demuestra un "Agente" utilizando el patrón ReAct (Reasoning + Acting).

✅ CARACTERÍSTICAS DE UN AGENTE:
   - Autonomía: El LLM recibe una pregunta y decide por sí mismo qué pasos tomar.
   - Uso de Herramientas: Puede "llamar" a funciones Python (buscar precios, noticias).
   - Bucle de Razonamiento: Si la primera búsqueda no es suficiente, puede hacer otra.

Conceptos Clave:
- Tools (Herramientas): Interfaces que conectan al LLM con el mundo exterior.
- Tool Calling: Capacidad nativa de modelos como GPT-4o para generar argumentos JSON estructurados.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import yfinance as yf

# Cargar claves
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: No se encontró la variable OPENAI_API_KEY en el archivo .env")
    print("   Por favor, crea un archivo .env y añade tu clave: OPENAI_API_KEY=sk-...")
    sys.exit(1)

# --- 1. DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---

@tool
def get_stock_price(symbol: str):
    """
    Obtiene el precio actual de una acción y su cambio porcentual.
    Input: Símbolo del ticker (ej: AAPL, MSFT, TSLA).
    """
    print(f"💰 Consultando precio para: {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            return "No se encontraron datos para este símbolo."
        price = data['Close'].iloc[0]
        return f"Precio actual de {symbol}: ${price:.2f}"
    except Exception as e:
        return f"Error en API financiera: {e}"

@tool
def search_financial_news(query: str):
    """
    Busca noticias financieras recientes en la web.
    Útil para entender el 'por qué' de un movimiento de mercado.
    """
    print(f"🌍 Buscando noticias sobre: {query}...")
    search = DuckDuckGoSearchRun()
    return search.run(query)

tools = [get_stock_price, search_financial_news]

# --- 2. EL CEREBRO (LLM) ---
# Usamos gpt-4o porque tiene excelente capacidad de tool calling
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 3. EL PROMPT DEL SISTEMA ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Eres un Analista de Inversiones Senior de Wall Street. 👔
    Tu trabajo es responder preguntas financieras con datos precisos y contexto.
    
    REGLAS:
    1. Siempre verifica el precio actual de la acción si se menciona una empresa.
    2. Busca noticias recientes para dar contexto (causas de subidas/bajadas).
    3. Sé conciso pero profesional. Usa tablas o bullet points si es necesario.
    """),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Aquí es donde el agente "piensa"
])

# --- 4. ENSAMBLAJE DEL AGENTE ---
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. INTERFAZ DE USUARIO ---
if __name__ == "__main__":
    print("--- 📈 AI INVESTMENT ANALYST ---")
    print("(Ejemplos: 'Analiza Apple', '¿Por qué bajó Tesla hoy?', 'Precio de Microsoft')")
    
    while True:
        query = input("\nPregunta al analista (o 'salir'): ")
        if query.lower() in ["salir", "exit"]:
            break
            
        try:
            response = agent_executor.invoke({"input": query})
            print("\n--- 📝 REPORTE FINAL ---")
            print(response["output"])
        except Exception as e:
            print(f"❌ Error: {e}")
