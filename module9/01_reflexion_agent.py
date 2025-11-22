"""
01_reflexion_agent.py
=====================
Implementación de un Agente de Reflexión (Reflexion) usando LangGraph.

Este agente intenta resolver una tarea de programación. Si falla (error de sintaxis o ejecución),
entra en un bucle de "Reflexión" donde analiza el error y propone una solución antes de reintentar.

Ciclo:
1.  **Draft:** Escribir código.
2.  **Execute:** Correr código.
3.  **Reflect:** Si falla -> Analizar traceback -> Guardar lección en memoria.
4.  **Retry:** Escribir nuevo código considerando la lección.

Requisitos:
pip install langgraph langchain langchain_openai
"""

from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- 1. Definición del Estado (Memoria del Grafo) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    code_solution: str
    error_log: str
    reflection: str
    iterations: int

# --- 2. Nodos del Grafo (Pasos del Pensamiento) ---

llm = ChatOpenAI(model="gpt-4", temperature=0)

def generate_code(state: AgentState):
    """Nodo Generador: Escribe la solución inicial o corregida."""
    print(f"✍️ Generando código (Iteración {state['iterations']})...")
    
    messages = state['messages']
    # Si hay reflexión previa, la inyectamos en el contexto
    if state.get('reflection'):
        messages.append(HumanMessage(content=f"Feedback anterior: {state['reflection']}. Por favor corrige el código."))
    
    # Simulamos la generación (en prod, usarías un prompt real de codificación)
    # Aquí hardcodeamos un error intencional en la primera vuelta para demo
    if state['iterations'] == 0:
        code = "print('Hola Mundo' + 5)" # Error de tipos
    else:
        code = "print('Hola Mundo' + ' 5')" # Corregido
        
    return {"code_solution": code, "iterations": state['iterations'] + 1}

def execute_code(state: AgentState):
    """Nodo Ejecutor: Corre el código y captura errores."""
    print("⚙️ Ejecutando código...")
    code = state['code_solution']
    try:
        exec(code)
        print("✅ Ejecución exitosa.")
        return {"error_log": ""}
    except Exception as e:
        print(f"❌ Error detectado: {e}")
        return {"error_log": str(e)}

def reflect_on_error(state: AgentState):
    """Nodo Reflexivo: Analiza por qué falló."""
    print("🧠 Reflexionando sobre el error...")
    error = state['error_log']
    code = state['code_solution']
    
    # El LLM analiza el error
    prompt = f"El código `{code}` falló con el error `{error}`. Explica brevemente por qué y cómo arreglarlo."
    reflection = llm.invoke(prompt).content
    
    print(f"💡 Insight: {reflection}")
    return {"reflection": reflection}

# --- 3. Construcción del Grafo (Wiring) ---

workflow = StateGraph(AgentState)

# Añadir Nodos
workflow.add_node("generate", generate_code)
workflow.add_node("execute", execute_code)
workflow.add_node("reflect", reflect_on_error)

# Definir Flujo
workflow.set_entry_point("generate")
workflow.add_edge("generate", "execute")

# Edge Condicional: ¿Hubo error?
def check_execution(state: AgentState):
    if state['error_log']:
        return "reflect" # Si hay error, ir a reflexionar
    return END           # Si no, terminar

workflow.add_conditional_edges(
    "execute",
    check_execution,
    {
        "reflect": "reflect",
        END: END
    }
)

workflow.add_edge("reflect", "generate") # Después de reflexionar, intentar de nuevo

# Compilar
app = workflow.compile()

# --- 4. Ejecución ---

if __name__ == "__main__":
    print("🚀 Iniciando Agente de Reflexión...")
    
    initial_state = {
        "messages": [HumanMessage(content="Escribe un script que sume texto y números.")],
        "iterations": 0,
        "code_solution": "",
        "error_log": "",
        "reflection": ""
    }
    
    # Ejecutar el grafo
    for event in app.stream(initial_state):
        pass # Los prints ya están en los nodos
