"""
Módulo 8 - Ejemplo Intermedio: Sistema de Debate Multi-Agente
Framework: AutoGen
Caso de uso: Análisis de decisiones de inversión

Múltiples agentes con perspectivas diferentes debaten y convergen
a una decisión consensuada.

Instalación:
pip install pyautogen python-dotenv
"""

import os
from typing import List
import autogen
from dotenv import load_dotenv

load_dotenv()


def create_debate_system():
    """Crear sistema de debate multi-agente con AutoGen"""
    
    # Configuración LLM
    llm_config = {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.7,
    }
    
    # Agente 1: Bull (Optimista)
    bull_agent = autogen.AssistantAgent(
        name="BullInvestor",
        system_message="""Eres un inversionista OPTIMISTA (Bull).
Tu rol es:
- Encontrar oportunidades y potencial de crecimiento
- Enfocarte en factores positivos y catalizadores de subida
- Defender por qué una inversión SUBIRÁ
- Ser entusiasta pero basarte en datos

Siempre firma tus mensajes como "- Bull Investor"
Sé conciso (máximo 4-5 líneas por mensaje).""",
        llm_config=llm_config,
    )
    
    # Agente 2: Bear (Pesimista)
    bear_agent = autogen.AssistantAgent(
        name="BearInvestor",
        system_message="""Eres un inversionista PESIMISTA (Bear).
Tu rol es:
- Identificar riesgos y señales de alerta
- Enfocarte en factores negativos y amenazas
- Defender por qué una inversión BAJARÁ o es riesgosa
- Ser escéptico pero basarte en datos

Siempre firma tus mensajes como "- Bear Investor"
Sé conciso (máximo 4-5 líneas por mensaje).""",
        llm_config=llm_config,
    )
    
    # Agente 3: Analyst (Neutral/Moderador)
    analyst_agent = autogen.AssistantAgent(
        name="Analyst",
        system_message="""Eres un ANALISTA NEUTRAL y moderador del debate.
Tu rol es:
- Sintetizar argumentos de Bull y Bear
- Hacer preguntas que profundicen el análisis
- Buscar consenso basado en evidencia
- Emitir recomendación final (Comprar/Vender/Hold)

Siempre firma tus mensajes como "- Analyst"
Sé conciso (máximo 4-5líneas por mensaje).""",
        llm_config=llm_config,
    )
    
    # User Proxy (inicia la conversación)
    user_proxy = autogen.UserProxyAgent(
        name="InvestmentCommittee",
        human_input_mode="NEVER",  # Automático
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    
    return bull_agent, bear_agent, analyst_agent, user_proxy


def run_investment_debate(company: str, bull: autogen.AssistantAgent, 
                          bear: autogen.AssistantAgent, 
                          analyst: autogen.AssistantAgent,
                          user_proxy: autogen.UserProxyAgent):
    """Ejecutar debate de inversión"""
    
    print("=" * 80)
    print(f"🎭 DEBATE DE INVERSIÓN: {company}")
    print("=" * 80)
    
    # Configurar group chat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, bull, bear, analyst],
        messages=[],
        max_round=12,  # Máximo 12 intercambios
        speaker_selection_method="round_robin",  # Turno rotativo
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0,
    })
    
    # Inicio del debate
    initial_message = f"""Analicemos la oportunidad de inversión en {company}.

Contexto reciente:
- La acción ha subido 25% en el último mes
- El P/E ratio está en 45 (industria promedio: 20)
- Reportaron crecimiento de ingresos del 40% YoY
- Hay preocupaciones sobre sostenibilidad a largo plazo

Bull Investor: Presenta el caso ALCISTA.
Bear Investor: Presenta el caso BAJISTA.
Analyst: Modera y sintetiza.

Al final, Analyst debe emitir una recomendación: COMPRAR, VENDER o MANTENER (HOLD)."""
    
    user_proxy.initiate_chat(
        manager,
        message=initial_message
    )
    
    # Extraer recomendación final
    messages = groupchat.messages
    final_recommendation = "No determinada"
    
    for msg in reversed(messages):
        if msg.get("name") == "Analyst":
            content = msg.get("content", "")
            if "COMPRAR" in content.upper():
                final_recommendation = "COMPRAR"
                break
            elif "VENDER" in content.upper():
                final_recommendation = "VENDER"
                break
            elif "HOLD" in content.upper() or "MANTENER" in content.upper():
                final_recommendation = "MANTENER"
                break
    
    print("\n" + "=" * 80)
    print(f"📊 RECOMENDACIÓN FINAL: {final_recommendation}")
    print("=" * 80)
    
    return final_recommendation


def main():
    """Demostración del sistema de debate"""
    
    print("=" * 80)
    print("Sistema de Debate Multi-Agente para Análisis de Inversiones")
    print("Powered by AutoGen")
    print("=" * 80)
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")    
    # Crear agentes
    bull, bear, analyst, user_proxy = create_debate_system()
    
    # Casos de estudio
    companies = [
        "NVIDIA (después del boom de IA)",
        # "Tesla (con la expansión de vehículos autónomos)",
        # "Una startup de IA generativa valuada en $1B"
    ]
    
    results = {}
    
    for company in companies:
        recommendation = run_investment_debate(company, bull, bear, analyst, user_proxy)
        results[company] = recommendation
        print("\n\n")
    
    # Resumen final
    print("=" * 80)
    print("📋 RESUMEN DE RECOMENDACIONES")
    print("=" * 80)
    for company, rec in results.items():
        print(f"   {company}: {rec}")
    print("=" * 80)
    
    print("\n✅ Debate completado. Los agentes han convergido a decisiones informadas.")


if __name__ == "__main__":
    main()
