"""
Módulo 7 - Ejemplo Avanzado: Tree-of-Thoughts con Beam Search
Framework: LangGraph
Caso de uso: Resolución de problemas matemáticos complejos

Tree-of-Thoughts explora múltiples caminos de razonamiento en paralelo,
evaluándolos y seleccionando los más prometedores (beam search).

Instalación:
pip install langgraph langchain langchain-openai
"""

import os
from typing import TypedDict, List, Dict, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o", temperature=0.8)  # Mayor temp para diversidad
EVALUATOR_LLM = ChatOpenAI(model="gpt-4o", temperature=0)  # Sin temp para evaluación


@dataclass
class ThoughtNode:
    """Nodo en el árbol de pensamiento"""
    id: int
    content: str  # El pensamiento/paso
    parent_id: int  # ID del nodo padre
    depth: int  # Profundidad en el árbol
    score: float  # Evaluación de calidad (0-10)
    is_solution: bool = False


class TreeOfThoughtsState(TypedDict):
    """Estado del sistema Tree-of-Thoughts"""
    problem: str
    max_depth: int
    beam_width: int  # Cuántos caminos mantener en cada nivel
    current_depth: int
    thought_tree: List[ThoughtNode]  # Todos los nodos
    active_nodes: List[int]  # IDs de nodos activos en la frontera
    best_solution: str
    next_node_id: int


def initialize_tree(state: TreeOfThoughtsState) -> TreeOfThoughtsState:
    """Paso 1: Inicializar árbol con nodo raíz"""
    print(f"\n🌳 Inicializando Tree-of-Thoughts")
    print(f"   Problema: {state['problem']}")
    print(f"   Beam width: {state['beam_width']}")
    print(f"   Max depth: {state['max_depth']}")
    
    # Nodo raíz: análisis inicial del problema
    root_prompt = f"""Analiza este problema y proporciona el primer paso de razonamiento:

Problema: {state['problem']}

Describe tu primer pensamiento o approach (1-2 líneas)."""
    
    response = LLM.invoke(root_prompt)
    root_thought = response.content.strip()
    
    root_node = ThoughtNode(
        id=0,
        content=root_thought,
        parent_id=-1,
        depth=0,
        score=5.0  # Neutral score inicial
    )
    
    print(f"\n🌱 Nodo raíz: {root_thought}")
    
    return {
        **state,
        "thought_tree": [root_node],
        "active_nodes": [0],
        "current_depth": 0,
        "next_node_id": 1
    }


def generate_thoughts(state: TreeOfThoughtsState) -> TreeOfThoughtsState:
    """Paso 2: Generar múltiples pensamientos desde cada nodo activo"""
    print(f"\n🌿 Generando pensamientos en profundidad {state['current_depth'] + 1}...")
    
    new_thoughts = []
    next_id = state["next_node_id"]
    
    # Generar 3 pensamientos por cada nodo activo
    for node_id in state["active_nodes"]:
        parent_node = next(n for n in state["thought_tree"] if n.id == node_id)
        
        # Construir contexto del camino hasta aquí
        path = get_path_to_node(state["thought_tree"], node_id)
        context = "\n".join([f"{i+1}. {n.content}" for i, n in enumerate(path)])
        
        generation_prompt = f"""Problema original: {state['problem']}

Razonamiento hasta ahora:
{context}

Genera 3 DIFERENTES próximos pasos de razonamiento. Cada uno debe explorar un approach distinto.
Sé creativo y considera múltiples posibilidades.

Formato:
OPCIÓN 1: [tu pensamiento]
OPCIÓN 2: [tu pensamiento distinto]
OPCIÓN 3: [tu pensamiento completamente diferente]"""
        
        response = LLM.invoke(generation_prompt)
        
        # Parsear las 3 opciones
        options = parse_options(response.content)
        
        for i, option in enumerate(options[:3], 1):  # Máximo 3 opciones
            new_node = ThoughtNode(
                id=next_id,
                content=option,
                parent_id=node_id,
                depth=parent_node.depth + 1,
                score=0.0  # Será evaluado después
            )
            new_thoughts.append(new_node)
            print(f"   💡 Opción {i} desde nodo {node_id}: {option[:60]}...")
            next_id += 1
    
    # Añadir nuevos pensamientos al árbol
    updated_tree = state["thought_tree"] + new_thoughts
    
    return {
        **state,
        "thought_tree": updated_tree,
        "next_node_id": next_id
    }


def evaluate_thoughts(state: TreeOfThoughtsState) -> TreeOfThoughtsState:
    """Paso 3: Evaluar calidad de los pensamientos generados"""
    print(f"\n⚖️ Evaluando pensamientos...")
    
    current_depth = state["current_depth"] + 1
    
    # Nodos a evaluar (los de la profundidad actual)
    nodes_to_eval = [n for n in state["thought_tree"] if n.depth == current_depth]
    
    evaluated_tree = state["thought_tree"].copy()
    
    for node in nodes_to_eval:
        # Construir camino completo
        path = get_path_to_node(state["thought_tree"], node.id)
        reasoning = "\n".join([f"{i+1}. {n.content}" for i, n in enumerate(path)])
        
        # Evaluar este camino de razonamiento
        eval_prompt = f"""Evalúa la calidad de este razonamiento para resolver el problema.

Problema: {state['problem']}

Razonamiento:
{reasoning}

Criterios:
- ¿Es lógico y coherente?
- ¿Avanza hacia la solución?
- ¿Es creativo/innovador?
- ¿Evita errores obvios?

Puntaje de 0-10 (solo responde con el número):"""
        
        response = EVALUATOR_LLM.invoke(eval_prompt)
        
        try:
            score = float(response.content.strip())
            score = max(0, min(10, score))  # Clamp entre 0-10
        except:
            score = 5.0  # Default si hay error
        
        # Actualizar score en el árbol
        for i, n in enumerate(evaluated_tree):
            if n.id == node.id:
                evaluated_tree[i] = ThoughtNode(
                    id=n.id,
                    content=n.content,
                    parent_id=n.parent_id,
                    depth=n.depth,
                    score=score
                )
                print(f"   📊 Nodo {n.id}: score = {score:.1f}")
                break
    
    return {**state, "thought_tree": evaluated_tree}


def select_best_beam(state: TreeOfThoughtsState) -> TreeOfThoughtsState:
    """Paso 4: Seleccionar top-K pensamientos (beam search)"""
    print(f"\n🎯 Seleccionando top-{state['beam_width']} pensamientos...")
    
    current_depth = state["current_depth"] + 1
    
    # Nodos de la profundidad actual
    current_nodes = [n for n in state["thought_tree"] if n.depth == current_depth]
    
    # Ordenar por score descendente
    sorted_nodes = sorted(current_nodes, key=lambda n: n.score, reverse=True)
    
    # Seleccionar top-K
    selected = sorted_nodes[:state["beam_width"]]
    selected_ids = [n.id for n in selected]
    
    print(f"   ✅ Seleccionados nodos: {selected_ids}")
    for node in selected:
        print(f"      - Nodo {node.id} (score: {node.score:.1f}): {node.content[:50]}...")
    
    return {
        **state,
        "active_nodes": selected_ids,
        "current_depth": current_depth
    }


def check_for_solution(state: TreeOfThoughtsState) -> TreeOfThoughtsState:
    """Paso 5: Verificar si algún camino llegó a una solución"""
    print(f"\n🔍 Verificando si hay soluciones completas...")
    
    best_score = 0
    best_solution = ""
    
    # Evaluar cada nodo activo
    for node_id in state["active_nodes"]:
        path = get_path_to_node(state["thought_tree"], node_id)
        reasoning = "\n".join([f"{i+1}. {n.content}" for i, n in enumerate(path)])
        
        # Preguntar si este camino es una solución completa
        solution_check_prompt = f"""Problema: {state['problem']}

Razonamiento completo:
{reasoning}

¿Este razonamiento resuelve COMPLETAMENTE el problema con una respuesta final?

Responde SOLO:
COMPLETE: [la respuesta final] (si está completo)
INCOMPLETE: [explicación breve] (si falta algo)"""
        
        response = EVALUATOR_LLM.invoke(solution_check_prompt)
        content = response.content.strip()
        
        if content.startswith("COMPLETE:"):
            solution = content.replace("COMPLETE:", "").strip()
            # Evaluar calidad de esta solución
            node = next(n for n in state["thought_tree"] if n.id == node_id)
            score = node.score
            
            print(f"   ✅ Solución encontrada en nodo {node_id} (score: {score:.1f})")
            
            if score > best_score:
                best_score = score
                best_solution = f"{reasoning}\n\nRESPUESTA FINAL: {solution}"
    
    if best_solution:
        print(f"\n🎉 Mejor solución encontrada!")
        return {**state, "best_solution": best_solution}
    
    return state


def should_continue(state: TreeOfThoughtsState) -> str:
    """Decisión: ¿Continuar explorando?"""
    # Parar si encontramos solución
    if state.get("best_solution"):
        return "end"
    
    # Parar si llegamos a profundidad máxima
    if state["current_depth"] >= state["max_depth"]:
        return "end"
    
    # Continuar explorando
    return "continue"


# Funciones auxiliares

def get_path_to_node(tree: List[ThoughtNode], node_id: int) -> List[ThoughtNode]:
    """Obtener camino desde la raíz hasta un nodo"""
    path = []
    current_id = node_id
    
    while current_id >= 0:
        node = next(n for n in tree if n.id == current_id)
        path.insert(0, node)
        current_id = node.parent_id
    
    return path


def parse_options(text: str) -> List[str]:
    """Parsear opciones del formato 'OPCIÓN 1: ...'"""
    options = []
    lines = text.split("\n")
    current_option = ""
    
    for line in lines:
        if line.startswith("OPCIÓN"):
            if current_option:
                options.append(current_option.strip())
            current_option = line.split(":", 1)[1] if ":" in line else ""
        elif current_option:
            current_option += " " + line
    
    if current_option:
        options.append(current_option.strip())
    
    return options if options else ["Continue reasoning...", "Try alternative approach...", "Consider edge cases..."]


def create_tree_of_thoughts_graph():
    """Construir grafo de Tree-of-Thoughts"""
    workflow = StateGraph(TreeOfThoughtsState)
    
    # Nodos
    workflow.add_node("initialize", initialize_tree)
    workflow.add_node("generate", generate_thoughts)
    workflow.add_node("evaluate", evaluate_thoughts)
    workflow.add_node("select", select_best_beam)
    workflow.add_node("check", check_for_solution)
    
    # Flujo
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", "select")
    workflow.add_edge("select", "check")
    
    # Loop o finalizar
    workflow.add_conditional_edges(
        "check",
        should_continue,
        {
            "continue": "generate",
            "end": END
        }
    )
    
    return workflow.compile()


def main():
    """Demostración de Tree-of-Thoughts"""
    print("=" * 80)
    print("Tree-of-Thoughts con Beam Search - Resolución de Problemas")
    print("=" * 80)
    
    # Problema complejo
    problem = """Tienes 8 bolas idénticas. Una de ellas pesa ligeramente diferente (no sabes si más o menos).
Tienes una balanza de dos platos. ¿Cuál es el MÍNIMO número de pesadas necesario para 
identificar la bola diferente Y determinar si pesa más o menos?"""
    
    # Crear sistema
    tot_system = create_tree_of_thoughts_graph()
    
    initial_state = TreeOfThoughtsState(
        problem=problem,
        max_depth=4,
        beam_width=2,  # Mantener 2 mejores caminos
        current_depth=0,
        thought_tree=[],
        active_nodes=[],
        best_solution="",
        next_node_id=0
    )
    
    # Ejecutar
    result = tot_system.invoke(initial_state)
    
    # Mostrar resultado
    print("\n" + "=" * 80)
    print("📝 SOLUCIÓN FINAL")
    print("=" * 80)
    
    if result["best_solution"]:
        print(result["best_solution"])
    else:
        print("No se encontró una solución completa en la profundidad permitida.")
        print("\nMejor camino explorado:")
        # Mostrar mejor camino
        best_node_id = max(result["active_nodes"], 
                          key=lambda nid: next(n.score for n in result["thought_tree"] if n.id == nid))
        path = get_path_to_node(result["thought_tree"], best_node_id)
        for i, node in enumerate(path, 1):
            print(f{i}. {node.content}")
    
    print(f"\n📊 Estadísticas:")
    print(f"   Total de pensamientos generados: {len(result['thought_tree'])}")
    print(f"   Profundidad alcanzada: {result['current_depth']}")
    print("=" * 80)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    main()
