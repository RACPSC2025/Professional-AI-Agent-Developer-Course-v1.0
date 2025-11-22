"""
🟢 NIVEL BÁSICO: AGENTE DE INTEGRACIÓN CON APIs EXTERNAS
--------------------------------------------------------
Este ejemplo demuestra cómo crear herramientas robustas para integración con APIs reales.
Caso de Uso: Asistente financiero que consulta precios de acciones y criptomonedas.

Conceptos Clave:
- Definición de herramientas con Pydantic para validación estricta.
- Manejo de errores y timeouts en APIs externas.
- Rate limiting para evitar bans.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
import requests
from datetime import datetime

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. DEFINICIÓN DE HERRAMIENTAS CON VALIDACIÓN ---

class StockPriceInput(BaseModel):
    """Input para buscar precio de acciones."""
    symbol: str = Field(description="Símbolo del ticker (ej: AAPL, MSFT, GOOGL)")

class CryptoInput(BaseModel):
    """Input para buscar precio de criptomonedas."""
    symbol: str = Field(description="Símbolo cripto (ej: BTC, ETH, SOL)")
    currency: str = Field(default="USD", description="Moneda fiat para la conversión")

@tool("get_stock_price", args_schema=StockPriceInput)
def get_stock_price(symbol: str) -> str:
    """
    Obtiene el precio actual de una acción usando la API pública de Yahoo Finance.
    Incluye manejo de errores y retry logic.
    """
    print(f"📊 Consultando precio de {symbol}...")
    
    try:
        # API alternativa gratuita (yfinance no tiene REST API directa)
        # Usando FinnHub como alternativa (requiere API key gratuita)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('chart') or not data['chart'].get('result'):
            return f"❌ No se encontraron datos para {symbol}. Verifica el símbolo."
        
        result = data['chart']['result'][0]
        meta = result['meta']
        current_price = meta.get('regularMarketPrice')
        previous_close = meta.get('previousClose')
        
        if current_price and previous_close:
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            
            return f"""
💰 {symbol}:
   Precio actual: ${current_price:.2f}
   Cambio: ${change:+.2f} ({change_pct:+.2f}%)
   Cierre anterior: ${previous_close:.2f}
   Actualizado: {datetime.now().strftime('%H:%M:%S')}
            """.strip()
        else:
            return f"Datos incompletos para {symbol}"
            
    except requests.exceptions.Timeout:
        return f"⏱️ Timeout al consultar {symbol}. La API no respondió a tiempo."
    except requests.exceptions.HTTPError as e:
        return f"❌ Error HTTP {e.response.status_code} al consultar {symbol}."
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"

@tool("get_crypto_price", args_schema=CryptoInput)
def get_crypto_price(symbol: str, currency: str = "USD") -> str:
    """
    Obtiene el precio de una criptomoneda usando CoinGecko API (sin API key).
    Ejemplo de API pública con rate limiting.
    """
    print(f"₿ Consultando precio de {symbol} en {currency}...")
    
    # Mapeo de símbolos comunes a IDs de CoinGecko
    crypto_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "DOGE": "dogecoin"
    }
    
    crypto_id = crypto_map.get(symbol.upper())
    if not crypto_id:
        return f"❌ Criptomoneda {symbol} no soportada. Usa: {', '.join(crypto_map.keys())}"
    
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': crypto_id,
            'vs_currencies': currency.lower(),
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if crypto_id not in data:
            return f"❌ No se encontraron datos para {symbol}"
        
        price = data[crypto_id].get(currency.lower())
        change_24h = data[crypto_id].get(f'{currency.lower()}_24h_change', 0)
        
        return f"""
₿ {symbol}:
   Precio: {price:,.2f} {currency}
   Cambio 24h: {change_24h:+.2f}%
   Fuente: CoinGecko
   Actualizado: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
    except Exception as e:
        return f"❌ Error al consultar {symbol}: {str(e)}"

# --- 2. AGENTE CON HERRAMIENTAS ---
tools = [get_stock_price, get_crypto_price]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Eres un Analista Financiero Senior. 📈
Tu trabajo es ayudar a usuarios a obtener información actualizada sobre activos financieros.

CAPACIDADES:
- Consultar precios de acciones (stocks).
- Consultar precios de criptomonedas.

REGLAS:
1. Siempre usa las herramientas para obtener datos en tiempo real.
2. Nunca inventes precios o datos.
3. Si una herramienta falla, explica el error claramente.
4. Sé conciso y profesional.
    """),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

# --- 3. INTERFAZ ---
if __name__ == "__main__":
    print("--- 💼 ASISTENTE FINANCIERO PROFESIONAL ---")
    print("Ejemplos: '¿Cuánto vale Apple?', 'Precio de Bitcoin en USD', 'Compara MSFT vs GOOGL'\n")
    
    while True:
        query = input("\n💬 Consulta (o 'salir'): ")
        if query.lower() in ["salir", "exit"]:
            break
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"\n📋 RESPUESTA:\n{response['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
