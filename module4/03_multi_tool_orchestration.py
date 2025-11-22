"""
🔴 NIVEL AVANZADO: ORQUESTACIÓN MULTI-HERRAMIENTA
-------------------------------------------------
Este ejemplo demuestra un agente que orquesta múltiples herramientas en paralelo y secuencialmente.
Caso de Uso: Sistema de Due Diligence automatizado para fusiones y adquisiciones (M&A).

Conceptos Clave:
- Ejecución paralela de herramientas para eficiencia.
- Composición de resultados de múltiples fuentes.
- Validación cruzada de datos.
"""

import os
import sys
import asyncio
from typing import Dict, List
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

# --- 1. HERRAMIENTAS ESPECIALIZADAS ---

class CompanyInput(BaseModel):
    """Input para consultas de empresa."""
    company_name: str = Field(description="Nombre de la empresa")

@tool("get_financial_metrics", args_schema=CompanyInput)
def get_financial_metrics(company_name: str) -> str:
    """
    Simula obtención de métricas financieras de una empresa.
    En producción, esto consultaría Bloomberg Terminal, SEC EDGAR, o similar.
    """
    print(f"📊 Consultando métricas financieras de {company_name}...")
    
    # Simulación de datos para demostración
    mock_data = {
        "apple": {
            "revenue": "394.3B USD",
            "ebitda": "123.1B USD",
            "debt_to_equity": "1.96",
            "current_ratio": "0.93",
            "roe": "147.4%"
        },
        "microsoft": {
            "revenue": "211.9B USD",
            "ebitda": "106.1B USD",
            "debt_to_equity": "0.48",
            "current_ratio": "1.77",
            "roe": "43.6%"
        }
    }
    
    company_key = company_name.lower()
    if company_key in mock_data:
        data = mock_data[company_key]
        return f"""
📈 Métricas Financieras de {company_name}:
   Ingresos: {data['revenue']}
   EBITDA: {data['ebitda']}
   Deuda/Capital: {data['debt_to_equity']}
   Ratio Corriente: {data['current_ratio']}
   ROE: {data['roe']}
   Fuente: Mock Data (En producción: Bloomberg/SEC)
        """.strip()
    else:
        return f"⚠️ Datos de {company_name} no disponibles en simulación. Use: Apple o Microsoft."

@tool("check_legal_issues", args_schema=CompanyInput)
def check_legal_issues(company_name: str) -> str:
    """
    Simula búsqueda de problemas legales y litigios pendientes.
    En producción, consultaría bases de datos legales (LexisNexis, Pacer, etc.).
    """
    print(f"⚖️ Verificando problemas legales de {company_name}...")
    
    # Simulación
    mock_legal = {
        "apple": {
            "active_lawsuits": 5,
            "regulatory_investigations": 2,
            "ip_disputes": 3,
            "risk_level": "Medium"
        },
        "microsoft": {
            "active_lawsuits": 3,
            "regulatory_investigations": 1,
            "ip_disputes": 1,
            "risk_level": "Low"
        }
    }
    
    company_key = company_name.lower()
    if company_key in mock_legal:
        data = mock_legal[company_key]
        return f"""
⚖️ Situación Legal de {company_name}:
   Demandas activas: {data['active_lawsuits']}
   Investigaciones regulatorias: {data['regulatory_investigations']}
   Disputas de PI: {data['ip_disputes']}
   Nivel de riesgo: {data['risk_level']}
   Fuente: Mock Data (En producción: LexisNexis)
        """.strip()
    else:
        return f"⚠️ Datos legales de {company_name} no disponibles."

@tool("get_market_sentiment", args_schema=CompanyInput)
def get_market_sentiment(company_name: str) -> str:
    """
    Simula análisis de sentimiento del mercado basado en noticias y redes sociales.
    En producción, usaría APIs de análisis de sentimiento (AlphaSense, Bloomberg Sentiment, etc.).
    """
    print(f"📰 Analizando sentimiento de mercado para {company_name}...")
    
    mock_sentiment = {
        "apple": {
            "sentiment_score": 0.72,
            "positive_mentions": "68%",
            "analyst_rating": "Buy",
            "price_target": "$195",
            "trending_topics": ["iPhone 16", "Vision Pro", "AI Integration"]
        },
        "microsoft": {
            "sentiment_score": 0.85,
            "positive_mentions": "78%",
            "analyst_rating": "Strong Buy",
            "price_target": "$450",
            "trending_topics": ["Azure AI", "GitHub Copilot", "Gaming"]
        }
    }
    
    company_key = company_name.lower()
    if company_key in mock_sentiment:
        data = mock_sentiment[company_key]
        return f"""
📰 Sentimiento de Mercado - {company_name}:
   Puntuación de sentimiento: {data['sentiment_score']}/1.0
   Menciones positivas: {data['positive_mentions']}
   Rating analistas: {data['analyst_rating']}
   Precio objetivo: {data['price_target']}
   Tendencias: {', '.join(data['trending_topics'])}
   Fuente: Mock Data (En producción: AlphaSense)
        """.strip()
    else:
        return f"⚠️ Datos de sentimiento de {company_name} no disponibles."

@tool("compare_competitors", args_schema=CompanyInput)
def compare_competitors(company_name: str) -> str:
    """
    Identifica y compara competidores directos de la empresa.
    """
    print(f"🔍 Comparando competidores de {company_name}...")
    
    mock_competitors = {
        "apple": {
            "main_competitors": ["Samsung", "Google", "Microsoft"],
            "market_share": "17.2%",
            "competitive_advantage": "Ecosistema cerrado, Brand loyalty"
        },
        "microsoft": {
            "main_competitors": ["Amazon (AWS)", "Google Cloud", "Salesforce"],
            "market_share": "21.5%",
            "competitive_advantage": "Enterprise integration, Azure AI"
        }
    }
    
    company_key = company_name.lower()
    if company_key in mock_competitors:
        data = mock_competitors[company_key]
        return f"""
🔍 Análisis Competitivo - {company_name}:
   Competidores principales: {', '.join(data['main_competitors'])}
   Cuota de mercado: {data['market_share']}
   Ventaja competitiva: {data['competitive_advantage']}
        """.strip()
    else:
        return f"⚠️ Análisis competitivo de {company_name} no disponible."

# --- 2. AGENTE DE ORQUESTACIÓN ---
tools = [
    get_financial_metrics,
    check_legal_issues,
    get_market_sentiment,
    compare_competitors
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Eres un Analista de M&A (Mergers & Acquisitions) Senior. 🏦

Tu misión es realizar Due Diligence exhaustivo de empresas usando múltiples herramientas.

METODOLOGÍA:
1. Para análisis completo, usa TODAS las herramientas disponibles.
2. Presenta los datos de forma estructurada y profesional.
3. Destaca riesgos y oportunidades.
4. Proporciona una recomendación final basada en datos.

CAPACIDADES:
- Métricas financieras (Ingresos, EBITDA, Ratios)
- Situación legal (Litigios, Riesgos regulatorios)
- Sentimiento de mercado (Noticias, Analistas)
- Análisis competitivo
    """),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10  # Permitimos más iteraciones para análisis completo
)

# --- 3. INTERFAZ ---
if __name__ == "__main__":
    print("="*60)
    print("  🏦 SISTEMA DE DUE DILIGENCE AUTOMATIZADO - M&A")
    print("="*60)
    print("\n📋 Este sistema orquesta múltiples herramientas para análisis completo.")
    print("   Empresas disponibles (demo): Apple, Microsoft\n")
    print("Ejemplos:")
    print("  - 'Realiza un análisis completo de Apple'")
    print("  - 'Compara las métricas financieras de Microsoft'")
    print("  - '¿Cuáles son los riesgos legales de Apple?'\n")
    
    while True:
        query = input("\n🔎 Consulta de Due Diligence (o 'salir'): ")
        if query.lower() in ["salir", "exit"]:
            print("\n👋 Sesión finalizada.")
            break
        
        try:
            print(f"\n⏳ Procesando análisis...")
            response = agent_executor.invoke({"input": query})
            
            print("\n" + "="*60)
            print("  📊 REPORTE DE DUE DILIGENCE")
            print("="*60)
            print(response['output'])
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error en el análisis: {e}")
