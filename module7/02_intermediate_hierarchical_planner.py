"""
Módulo 7 - Ejemplo Intermedio: Planificador Jerárquico
Framework: LangGraph
Caso de uso: Planificador de eventos corporativos

Planificación jerárquica descompone problemas complejos en sub-tareas
y las ejecuta en orden, adaptándose a resultados previos.

Instalación:
pip install langgraph langchain langchain-openai
"""

import os
from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class PlannerState(TypedDict):
    """Estado del planificador jerárquico"""
    goal: str  # Objetivo principal
    plan: List[Dict]  # Plan jerárquico
    current_step: int  # Paso actual
    completed_tasks: List[str]  # Tareas completadas
    context: Dict  # Context acumulado
    final_result: str


def create_hierarchical_plan(state: PlannerState) -> PlannerState:
    """Paso 1: Crear plan jerárquico descomponiendo el objetivo"""
    print(f"\n🎯 Objetivo: {state['goal']}")
    print("\n📝 Creando plan jerárquico...")
    
    planning_prompt = f"""Eres un planificador experto de eventos corporativos.

Objetivo: {state['goal']}

Descompón este objetivo en un plan jerárquico con las siguientes fases:
1. INVESTIGACIÓN: Recopilar información necesaria
2. DISEÑO: Planear los detalles específicos
3. LOGÍSTICA: Organizar recursos y proveedores
4. EJECUCIÓN: Steps de implementación

Para cada fase, especifica 2-3 sub-tareas concretas.

Formato de respuesta (JSON):
[
  {{"phase": "INVESTIGACIÓN", "task": "Identificar requisitos del cliente", "dependencies": []}},
  {{"phase": "INVESTIGACIÓN", "task": "Analizar presupuesto disponible", "dependencies": []}},
  ...
]

Solo responde con el JSON, sin texto adicional."""
    
    response = LLM.invoke(planning_prompt)
    
    # Parsear respuesta (simplificado)
    import json
    try:
        plan = json.loads(response.content)
    except:
        # Fallback si el parsing falla
        plan = [
            {"phase": "INVESTIGACIÓN", "task": "Identificar requisitos del cliente", "dependencies": []},
            {"phase": "INVESTIGACIÓN", "task": "Analizar presupuesto disponible", "dependencies": []},
            {"phase": "DISEÑO", "task": "Seleccionar venue y fecha", "dependencies": ["Identificar requisitos del cliente"]},
            {"phase": "DISEÑO", "task": "Diseñar agenda del evento", "dependencies": ["Identificar requisitos del cliente"]},
            {"phase": "LOGÍSTICA", "task": "Contratar catering", "dependencies": ["Analizar presupuesto disponible", "Seleccionar venue y fecha"]},
            {"phase": "LOGÍSTICA", "task": "Reservar audiovisuales", "dependencies": ["Seleccionar venue y fecha"]},
            {"phase": "EJECUCIÓN", "task": "Enviar invitaciones", "dependencies": ["Diseñar agenda del evento"]},
            {"phase": "EJECUCIÓN", "task": "Coordinar setup del día", "dependencies": ["Contratar catering", "Reservar audiovisuales"]},
        ]
    
    print(f"✅ Plan creado con {len(plan)} tareas:")
    for i, step in enumerate(plan, 1):
        deps = f" (depende de: {', '.join(step['dependencies'])})" if step['dependencies'] else ""
        print(f"   {i}. [{step['phase']}] {step['task']}{deps}")
    
    return {**state, "plan": plan, "current_step": 0}


def execute_task(state: PlannerState) -> PlannerState:
    """Paso 2: Ejecutar tarea actual"""
    current_idx = state["current_step"]
    
    if current_idx >= len(state["plan"]):
        # Todas las tareas completadas
        return state
    
    current_task = state["plan"][current_idx]
    
    print(f"\n⚙️ Ejecutando: [{current_task['phase']}] {current_task['task']}")
    
    # Verificar dependencias
    dependencies = current_task.get("dependencies", [])
    completed = state["completed_tasks"]
    
    for dep in dependencies:
        if dep not in completed:
            print(f"   ⏸️ Esperando dependencia: {dep}")
            # En un sistema real, esto manejaría el reordenamiento
            return state
    
    # Recopilar contexto de tareas previas
    previous_context = "\n".join([
        f"- {task}: {state['context'].get(task, 'N/A')}" 
        for task in completed
    ])
    
    # Ejecutar tarea
    execution_prompt = f"""Objetivo general: {state['goal']}

Tarea actual: {current_task['task']}
Fase: {current_task['phase']}

Contexto de tareas previas:
{previous_context if previous_context else "Ninguna (primera tarea)"}

Ejecuta esta tarea y proporciona:
1. El resultado concreto de completar esta tarea
2. Información relevante que otras tareas puedan necesitar

Sé específico y conciso (2-3 líneas)."""
    
    response = LLM.invoke(execution_prompt)
    result = response.content.strip()
    
    print(f"   ✅ Completado: {result[:100]}...")
    
    # Actualizar estado
    new_completed = completed + [current_task['task']]
    new_context = {**state["context"], current_task['task']: result}
    
    return {
        **state,
        "current_step": current_idx + 1,
        "completed_tasks": new_completed,
        "context": new_context
    }


def review_progress(state: PlannerState) -> PlannerState:
    """Paso 3: Revisar progreso y adaptar si es necesario"""
    completed_count = len(state["completed_tasks"])
    total_count = len(state["plan"])
    
    print(f"\n📊 Progreso: {completed_count}/{total_count} tareas completadas")
    
    if completed_count == total_count:
        print("🎉 ¡Todas las tareas completadas!")
    
    return state


def synthesize_final_result(state: PlannerState) -> PlannerState:
    """Paso 4: Sintetizar resultado final"""
    print("\n🎨 Sintetizando resultado final...")
    
    # Recopilar todos los resultados
    all_results = "\n".join([
        f"{task}: {result}" 
        for task, result in state["context"].items()
    ])
    
    synthesis_prompt = f"""Basándote en todas las tareas completadas, genera un resumen ejecutivo 
del plan completo para: {state['goal']}

Tareas completadas y resultados:
{all_results}

Crea un resumen estructurado con:
1. Resumen ejecutivo (2-3 líneas)
2. Cronograma estimado
3. Presupuesto estimado
4. Próximos pasos

Formato profesional y conciso."""
    
    response = LLM.invoke(synthesis_prompt)
    final_result = response.content
    
    print(f"\n📄 Resultado Final:")
    print("=" * 70)
    print(final_result)
    print("=" * 70)
    
    return {**state, "final_result": final_result}


def should_continue(state: PlannerState) -> str:
    """Decisión: ¿Continuar ejecutando tareas?"""
    if state["current_step"] < len(state["plan"]):
        return "execute"
    else:
        return "synthesize"


def create_hierarchical_planner():
    """Construir grafo de planificador jerárquico"""
    workflow = StateGraph(PlannerState)
    
    # Nodos
    workflow.add_node("plan", create_hierarchical_plan)
    workflow.add_node("execute", execute_task)
    workflow.add_node("review", review_progress)
    workflow.add_node("synthesize", synthesize_final_result)
    
    # Flujo
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    
    # Loop de ejecución
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "execute": "review",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("review", "execute")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


def main():
    """Demostración de planificador jerárquico"""
    print("=" * 70)
    print("Planificador Jerárquico de Eventos Corporativos")
    print("=" * 70)
    
    # Objetivo del evento
    goal = "Organizar una conferencia tech de 100 personas sobre IA en 3 meses"
    
    # Crear y ejecutar planificador
    planner = create_hierarchical_planner()
    
    initial_state = PlannerState(
        goal=goal,
        plan=[],
        current_step=0,
        completed_tasks=[],
        context={},
        final_result=""
    )
    
    # Ejecutar (esto hará múltiples iteraciones)
    result = planner.invoke(initial_state)
    
    print("\n✅ Planificación completada exitosamente!")
    print(f"\n📋 Total de tareas ejecutadas: {len(result['completed_tasks'])}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    main()
