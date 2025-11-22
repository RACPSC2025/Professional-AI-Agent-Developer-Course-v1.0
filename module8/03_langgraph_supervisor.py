"""
03_langgraph_supervisor.py
==========================
Este script implementa el patrón "Supervisor" usando LangGraph.
Un agente central (Supervisor) actúa como un router inteligente, dirigiendo el trabajo
a agentes especializados (Workers) y agregando sus resultados.

Caso de Uso: Sistema de Soporte al Cliente que enruta a Facturación o Soporte Técnico.

Requisitos:
pip install langgraph langchain langchain-openai
"""

import operator
from typing import Annotated, List, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph, END

# 1. Definir el Estado del Grafo
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# 2. Definir los Workers (Agentes Especializados)

def billing_agent(state: AgentState):
    print("💰 Billing Agent: Procesando consulta de facturación...")
    return {"messages": [AIMessage(content="He revisado tu factura. El cargo de $50 es correcto.")]}

def tech_support_agent(state: AgentState):
    print("🔧 Tech Support: Procesando consulta técnica...")
    return {"messages": [AIMessage(content="Para reiniciar el router, mantén presionado el botón 10 segundos.")]}

# 3. Definir el Supervisor (Router)
# El supervisor usa Function Calling (Structured Output) para decidir quién sigue.

class RouteResponse(BaseModel):
    next: Literal["Billing", "TechSupport", "FINISH"]

llm = ChatOpenAI(model="gpt-3.5-turbo")

def supervisor_node(state: AgentState):
    print("👔 Supervisor: Enrutando solicitud...")
    
    system_prompt = """Eres un supervisor de soporte.
    Tu trabajo es enrutar la consulta del usuario al departamento correcto:
    - Billing: Para pagos, facturas, dinero.
    - TechSupport: Para problemas técnicos, errores, configuración.
    - FINISH: Si la consulta ya fue respondida o es un saludo simple.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Decide quién debe actuar ahora: Billing, TechSupport, o FINISH.")
    ])
    
    # Usamos with_structured_output para forzar una decisión válida
    chain = prompt | llm.with_structured_output(RouteResponse)
    
    decision = chain.invoke({"messages": state["messages"]})
    return {"next": decision.next}

# 4. Construir el Grafo
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Billing", billing_agent)
workflow.add_node("TechSupport", tech_support_agent)

workflow.set_entry_point("Supervisor")

# Conditional Edges
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "Billing": "Billing",
        "TechSupport": "TechSupport",
        "FINISH": END
    }
)

# Los workers siempre reportan de vuelta al Supervisor
workflow.add_edge("Billing", "Supervisor")
workflow.add_edge("TechSupport", "Supervisor")

app = workflow.compile()

# 5. Ejecutar
def main():
    # Caso 1: Facturación
    print("--- Caso 1: Facturación ---")
    inputs = {"messages": [HumanMessage(content="Tengo un cargo extraño en mi tarjeta de crédito.")]}
    for s in app.stream(inputs):
        pass # Logs en nodos
        
    print("\n--- Caso 2: Técnico ---")
    inputs = {"messages": [HumanMessage(content="Mi internet no funciona.")]}
    for s in app.stream(inputs):
        pass

if __name__ == "__main__":
    main()
