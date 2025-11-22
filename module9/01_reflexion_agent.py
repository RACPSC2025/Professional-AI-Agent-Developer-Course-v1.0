"""
01_reflexion_agent.py
=====================
Implementación de un Agente Reflexion usando LangGraph.
Este agente intenta resolver una tarea de programación. Si su código falla,
analiza el error (Reflexion), actualiza su memoria y reintenta.

Ciclo:
Generator -> Executor -> (Error) -> Reflector -> Generator
                      -> (Success) -> END

Requisitos:
pip install langgraph langchain langchain-openai
"""

from typing import List, TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Configuración
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 1. Definir Estado
class ReflexionState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    code: str
    iterations: int
    error: str
    success: bool

# 2. Nodos

def generator_node(state: ReflexionState):
    """Genera o corrige el código basado en el historial"""
    print(f"🤖 Generator: Escribiendo código (Iteración {state['iterations']})...")
    
    messages = state['messages']
    
    # Si hay error previo, el reflector ya añadió el contexto, así que el LLM lo verá
    response = llm.invoke(messages)
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "code": code,
        "messages": [response],
        "iterations": state['iterations'] + 1
    }

def executor_node(state: ReflexionState):
    """Simula la ejecución del código"""
    code = state['code']
    print("⚡ Executor: Ejecutando código...")
    
    try:
        # PELIGRO: exec() es inseguro en prod. Usar sandbox (e2b, docker) en realidad.
        # Aquí simulamos un entorno seguro con variables locales
        local_scope = {}
        exec(code, {}, local_scope)
        
        # Verificación simple (asumimos que el código debe definir una función 'solve')
        if "solve" not in local_scope:
            raise Exception("El código debe definir una función llamada 'solve'")
            
        # Test simple
        result = local_scope["solve"]()
        print(f"   Resultado: {result}")
        
        return {"success": True, "error": ""}
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {"success": False, "error": str(e)}

def reflector_node(state: ReflexionState):
    """Analiza el error y genera feedback constructivo"""
    print("🧠 Reflector: Analizando por qué falló...")
    
    error = state['error']
    code = state['code']
    
    prompt = f"""
    Tu código anterior falló con este error: "{error}"
    
    Código:
    {code}
    
    Analiza el error y da instrucciones precisas para corregirlo.
    Sé breve y técnico.
    """
    
    reflection = llm.invoke([HumanMessage(content=prompt)])
    
    # Añadimos la reflexión como mensaje del usuario para que el Generator la vea
    feedback_msg = HumanMessage(content=f"El código falló: {error}. Consejo: {reflection.content}")
    
    return {"messages": [feedback_msg]}

# 3. Grafo
workflow = StateGraph(ReflexionState)

workflow.add_node("generator", generator_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("generator")

workflow.add_edge("generator", "executor")

def should_continue(state: ReflexionState):
    if state['success']:
        return "end"
    if state['iterations'] > 3: # Límite de intentos
        return "end"
    return "reflect"

workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "end": END,
        "reflect": "reflector"
    }
)

workflow.add_edge("reflector", "generator")

app = workflow.compile()

# 4. Ejecutar
def main():
    task = """
    Escribe una función en Python llamada 'solve' que retorne la suma de los primeros 10 números primos.
    Asegúrate de importar lo necesario.
    """
    
    print(f"Tarea: {task}\n")
    
    inputs = {
        "messages": [HumanMessage(content=task)],
        "iterations": 0,
        "success": False,
        "error": "",
        "code": ""
    }
    
    for event in app.stream(inputs):
        pass # Logs en nodos

if __name__ == "__main__":
    main()
