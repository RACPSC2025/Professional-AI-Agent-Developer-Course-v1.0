"""
01_plan_and_execute.py
======================
Este script implementa el patrón "Plan-and-Execute" utilizando LangGraph.
A diferencia de un agente ReAct que decide paso a paso, este agente primero crea un plan completo
y luego lo ejecuta secuencialmente. Esto reduce errores en tareas complejas.

Arquitectura:
1. Planner Node: LLM que genera una lista de pasos (List[str]).
2. Executor Node: Agente que ejecuta el paso actual.
3. State: Mantiene el plan, pasos completados y resultados.

Requisitos:
pip install langgraph langchain langchain-openai
"""

import operator
from typing import Annotated, List, TypedDict, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# 1. Definir el Estado del Grafo
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], operator.add]
    response: str

# 2. Definir el Modelo del Plan (Structured Output)
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")

# 3. Nodos del Grafo

def planner_node(state: PlanExecuteState):
    """Genera el plan inicial"""
    print("🧠 Planner: Generando plan...")
    model = ChatOpenAI(model="gpt-3.5-turbo").with_structured_output(Plan)
    
    prompt = ChatPromptTemplate.from_template(
        "Para el siguiente objetivo, genera un plan paso a paso simple: {objective}"
    )
    
    chain = prompt | model
    plan = chain.invoke({"objective": state["input"]})
    
    return {"plan": plan.steps}

def executor_node(state: PlanExecuteState):
    """Ejecuta el primer paso del plan"""
    plan = state["plan"]
    if not plan:
        return {"response": "Plan completado"}
    
    current_step = plan[0]
    print(f"🛠️ Executor: Ejecutando paso -> {current_step}")
    
    # Simulación de ejecución (aquí irían llamadas a tools reales)
    # En un caso real, usaríamos un agente ReAct aquí para resolver el paso
    result = f"Resultado simulado para: {current_step}"
    
    return {
        "past_steps": [(current_step, result)],
        "plan": plan[1:] # Eliminar paso completado
    }

def should_end(state: PlanExecuteState):
    """Decide si terminar o seguir ejecutando"""
    if not state["plan"]:
        return "end"
    return "continue"

def response_node(state: PlanExecuteState):
    """Genera la respuesta final basada en los pasos ejecutados"""
    print("✅ Generando respuesta final...")
    # Aquí un LLM sintetizaría todos los past_steps
    return {"response": "Tarea finalizada con éxito basada en los pasos ejecutados."}

# 4. Construir el Grafo
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("responder", response_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")

workflow.add_conditional_edges(
    "executor",
    should_end,
    {
        "continue": "executor",
        "end": "responder"
    }
)

workflow.add_edge("responder", END)

app = workflow.compile()

# 5. Ejecutar
def main():
    inputs = {"input": "Investiga el precio de las acciones de Tesla, compáralo con Ford y escribe un resumen."}
    
    print(f"Objetivo: {inputs['input']}\n")
    
    for event in app.stream(inputs):
        for key, value in event.items():
            pass # Ya imprimimos logs en los nodos

if __name__ == "__main__":
    main()
