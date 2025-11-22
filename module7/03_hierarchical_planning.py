"""
03_hierarchical_planning.py
===========================
Implementación de un sistema de planificación jerárquica (Supervisor-Worker).
Un agente "Supervisor" descompone una tarea y delega a agentes "Workers" especializados.

Arquitectura:
- Supervisor: LLM que decide a quién llamar o si terminar.
- Workers: Researcher, Coder.
- Grafo: Supervisor -> [Researcher, Coder] -> Supervisor

Requisitos:
pip install langgraph langchain langchain-openai
"""

from typing import Annotated, List, TypedDict, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import operator

# 1. Estado del Grafo
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str

# 2. Agentes Workers
llm = ChatOpenAI(model="gpt-3.5-turbo")

def researcher_node(state: AgentState):
    print("🔎 Researcher: Buscando información...")
    last_message = state["messages"][-1]
    # Simulación de research
    response = f"Research results for: {last_message.content}. Found relevant data."
    return {"messages": [AIMessage(content=response)]}

def coder_node(state: AgentState):
    print("💻 Coder: Escribiendo código...")
    last_message = state["messages"][-1]
    # Simulación de coding
    response = f"Python code generated based on: {last_message.content}"
    return {"messages": [AIMessage(content=response)]}

# 3. Agente Supervisor (Router)
def supervisor_node(state: AgentState):
    print("👔 Supervisor: Decidiendo siguiente paso...")
    
    # El supervisor analiza la conversación y decide quién sigue
    # En un caso real, usamos function calling para esto
    
    last_message = state["messages"][-1]
    content = last_message.content.lower()
    
    if "research" in content and "code" not in content:
        return {"next_agent": "coder"} # Después de research, toca code
    elif "code" in content:
        return {"next_agent": "FINISH"} # Después de code, terminamos
    else:
        return {"next_agent": "researcher"} # Empezamos con research

# 4. Construcción del Grafo
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

workflow.set_entry_point("supervisor")

# Conditional Edges desde Supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_agent"],
    {
        "researcher": "researcher",
        "coder": "coder",
        "FINISH": END
    }
)

# Workers siempre vuelven a Supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder", "supervisor")

app = workflow.compile()

# 5. Ejecutar
def main():
    task = "Investiga sobre RAG y escribe un script de ejemplo."
    print(f"Tarea: {task}\n")
    
    inputs = {"messages": [HumanMessage(content=task)]}
    
    for event in app.stream(inputs):
        pass # Logs en nodos

if __name__ == "__main__":
    main()
