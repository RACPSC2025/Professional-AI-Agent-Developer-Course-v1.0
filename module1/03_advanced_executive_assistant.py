"""
🔴 NIVEL AVANZADO: ASISTENTE EJECUTIVO (THE SYSTEM)
---------------------------------------------------
Este script simula un sistema agéntico capaz de realizar acciones de escritura (Side Effects).

⚠️ CRÍTICO - SEGURIDAD EN AGENTES:
   - Human-in-the-loop: NUNCA dejes que un agente envíe emails o borre archivos sin confirmación.
   - Principio de Mínimo Privilegio: El agente solo debe tener acceso a lo estrictamente necesario.
   - OAuth: En producción, usa tokens de usuario (no Service Accounts) para actuar en nombre de una persona.

Conceptos Clave:
- Write Actions: El agente MODIFICA el estado del mundo (crea eventos, envía emails).
- Mocking: Simulamos las herramientas para aprender sin riesgo.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from google.ai.generativelanguage_v1beta.types import content

# Cargar API Key
load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Descomentar si tienes la key

# --- 1. HERRAMIENTAS SIMULADAS (MOCKS) ---

# Base de datos simulada
calendar_db = {
    "lunes": ["09:00 - Daily Meeting", "14:00 - Almuerzo con Cliente"],
    "martes": ["10:00 - Revisión de Código"]
}

def list_calendar_events(day: str):
    """Lista los eventos del calendario para un día específico (lunes, martes, etc)."""
    print(f"📅 [API] Leyendo calendario para: {day}...")
    return calendar_db.get(day.lower(), ["No hay eventos."])

def send_email(recipient: str, subject: str, body: str):
    """Envía un correo electrónico real (Simulado)."""
    print(f"📧 [API] ENVIANDO EMAIL a {recipient}")
    print(f"   Asunto: {subject}")
    print(f"   Cuerpo: {body[:50]}...")
    return "Email enviado exitosamente con ID: #5521"

# Diccionario de herramientas para Gemini
tools_map = {
    'list_calendar_events': list_calendar_events,
    'send_email': send_email
}

# --- 2. CONFIGURACIÓN DEL MODELO (Simulación de Lógica) ---
# Nota: En un entorno real con Google SDK, pasarías `tools=[list_calendar_events, send_email]`
# al constructor del modelo. Aquí simulamos el bucle de decisión para fines educativos
# si no tienes la API Key de Google configurada.

def simulated_agent_loop(user_query):
    print(f"\n🤖 AGENTE RECIBIÓ: '{user_query}'")
    
    # Paso 1: El modelo "piensa" (Hardcoded para demostración)
    if "reunión" in user_query and "lunes" in user_query:
        # El modelo decide llamar a la herramienta de lectura
        tool_name = "list_calendar_events"
        tool_args = {"day": "lunes"}
        
        # Paso 2: Ejecución de Herramienta
        result = tools_map[tool_name](**tool_args)
        
        # Paso 3: Razonamiento sobre el resultado
        print(f"🤔 Agente: Veo que el usuario tiene: {result}")
        
        if "Daily Meeting" in str(result):
            print("🤔 Agente: Hay conflicto a las 09:00. Debo avisar.")
            
            # Paso 4: Acción de Escritura (Enviar Email)
            email_tool = "send_email"
            email_args = {
                "recipient": "jefe@empresa.com", 
                "subject": "Conflicto de Agenda", 
                "body": "Hola, no podré asistir a la Daily porque tengo..."
            }
            final_result = tools_map[email_tool](**email_args)
            return f"He revisado tu agenda y enviado un correo de aviso. ({final_result})"
            
    return "No estoy seguro de qué hacer."

# --- 3. EJECUCIÓN ---
if __name__ == "__main__":
    print("--- 👔 AI EXECUTIVE ASSISTANT (MOCK) ---")
    print("Este agente tiene permiso para LEER tu calendario y ENVIAR emails.")
    
    query = "Revisa si tengo alguna reunión el lunes por la mañana y si es así avisa a mi jefe."
    response = simulated_agent_loop(query)
    
    print(f"\n✅ RESPUESTA FINAL: {response}")
