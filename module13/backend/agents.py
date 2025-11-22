"""
backend/agents.py
=================
Definición del Grafo de Agentes (The Software House).
Usa LangGraph para orquestar el flujo PM -> Coder -> Reviewer.
"""

from typing import TypedDict, List, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- 1. Estado del Equipo ---
class TeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    requirement: str
    plan: str
    code: str
    review_comments: str
    iteration: int

# --- 2. Los Agentes (Nodos) ---
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def product_manager(state: TeamState):
    print("👔 PM: Analizando requerimientos...")
    req = state['requirement']
    prompt = f"Eres un Product Manager experto. Crea un plan técnico paso a paso para: {req}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"plan": response.content, "messages": [response]}

def senior_coder(state: TeamState):
    print("👨‍💻 Coder: Escribiendo código...")
    plan = state['plan']
    comments = state.get('review_comments', '')
    
    prompt = f"""Eres un Senior Python Developer.
    Plan: {plan}
    Feedback anterior (si hay): {comments}
    
    Escribe el código Python completo para implementar esto. Solo el código."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"code": response.content, "messages": [response]}

def qa_engineer(state: TeamState):
    print("🧐 QA: Revisando código...")
    code = state['code']
    prompt = f"""Revisa el siguiente código Python. 
    Si es correcto y seguro, responde solo con 'APROBADO'.
    Si tiene errores, lístalos.
    
    Código:
    {code}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"review_comments": response.content, "messages": [response], "iteration": state['iteration'] + 1}

# --- 3. Lógica de Control (Edges) ---
def should_continue(state: TeamState):
    review = state['review_comments']
    iteration = state['iteration']
    
    if "APROBADO" in review:
        return "end"
    
    if iteration > 3: # Límite de intentos para evitar bucles infinitos
        return "end"
        
    return "retry"

# --- 4. Construcción del Grafo ---
workflow = StateGraph(TeamState)

workflow.add_node("pm", product_manager)
workflow.add_node("coder", senior_coder)
workflow.add_node("qa", qa_engineer)

workflow.set_entry_point("pm")
workflow.add_edge("pm", "coder")
workflow.add_edge("coder", "qa")

workflow.add_conditional_edges(
    "qa",
    should_continue,
    {
        "retry": "coder",
        "end": END
    }
)

app = workflow.compile()
