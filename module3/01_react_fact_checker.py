"""
🟢 NIVEL BÁSICO: VERIFICADOR DE HECHOS (PATRÓN REACT)
-----------------------------------------------------
Este script implementa el patrón ReAct (Reason + Act) DESDE CERO.
No usamos LangChain ni AutoGen aquí para que entiendas la lógica interna.

El ciclo es:
1. PENSAMIENTO: El LLM analiza qué necesita saber.
2. ACCIÓN: El LLM elige una herramienta (ej: buscar en Wikipedia).
3. OBSERVACIÓN: El código ejecuta la herramienta y le da el resultado al LLM.
4. REPETIR: Hasta que el LLM tenga suficiente info para responder.

Caso de Uso: Verificar afirmaciones complejas que requieren múltiples pasos.
"""

import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# --- 1. HERRAMIENTAS ---
search_tool = DuckDuckGoSearchRun()

def execute_tool(tool_name, tool_input):
    if tool_name == "SEARCH":
        print(f"   🔍 BUSCANDO: {tool_input}...")
        try:
            return search_tool.run(tool_input)
        except Exception as e:
            return f"Error en búsqueda: {e}"
    return "Herramienta no encontrada."

# --- 2. PROMPT DEL SISTEMA (EL CEREBRO REACT) ---
REACT_SYSTEM_PROMPT = """
Eres un Verificador de Hechos experto. Tu trabajo es validar afirmaciones.
Para responder, DEBES usar el siguiente formato:

Pensamiento: [Tu razonamiento sobre qué hacer ahora]
Acción: [SEARCH]
Entrada de Acción: [Lo que quieres buscar]

Cuando tengas la respuesta final:
Pensamiento: [Ya tengo la respuesta]
Respuesta Final: [Tu conclusión veraz]

Ejemplo:
Pregunta: ¿Quién es el CEO de Apple?
Pensamiento: Debo buscar quién es el CEO actual.
Acción: SEARCH
Entrada de Acción: CEO actual de Apple
... (El sistema te dará la Observación) ...
Pensamiento: La búsqueda dice que es Tim Cook.
Respuesta Final: El CEO de Apple es Tim Cook.

¡EMPIEZA!
"""

# --- 3. EL BUCLE REACT (EL MOTOR) ---
def run_react_agent(question, max_steps=5):
    print(f"🤖 PREGUNTA: {question}\n")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    history = [
        ("system", REACT_SYSTEM_PROMPT),
        ("user", f"Pregunta: {question}")
    ]
    
    step = 0
    while step < max_steps:
        step += 1
        print(f"--- PASO {step} ---")
        
        # 1. LLM Genera Pensamiento + Acción
        response = llm.invoke(history).content
        print(f"🧠 AGENTE:\n{response}")
        history.append(("assistant", response))
        
        # 2. Detectar si hay Respuesta Final
        if "Respuesta Final:" in response:
            return response.split("Respuesta Final:")[1].strip()
            
        # 3. Parsear Acción (Regex simple)
        # Buscamos: Acción: SEARCH \n Entrada de Acción: query
        action_match = re.search(r"Acción:\s*(\w+)", response)
        input_match = re.search(r"Entrada de Acción:\s*(.+)", response)
        
        if action_match and input_match:
            tool_name = action_match.group(1)
            tool_input = input_match.group(1).strip()
            
            # 4. Ejecutar Herramienta
            observation = execute_tool(tool_name, tool_input)
            print(f"👀 OBSERVACIÓN: {observation[:200]}...") # Truncamos para no ensuciar log
            
            # 5. Alimentar de vuelta al LLM
            history.append(("user", f"Observación: {observation}"))
        else:
            print("⚠️ El agente no generó una acción válida. Forzando continuación...")
            history.append(("user", "Por favor sigue el formato: Acción: [TOOL] / Entrada de Acción: [INPUT]"))

    return "❌ Se alcanzó el límite de pasos sin respuesta."

# --- 4. EJECUCIÓN ---
if __name__ == "__main__":
    # Pregunta trampa: Requiere saber quién inventó el transistor Y si ganó 2 Nobels
    q = "¿Es verdad que el inventor del transistor ganó dos premios Nobel?"
    resultado = run_react_agent(q)
    print(f"\n✅ RESULTADO FINAL: {resultado}")
