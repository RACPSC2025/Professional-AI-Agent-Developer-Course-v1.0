"""
Módulo 9 - Ejemplo Intermedio: Agente Reflexion con Auto-Mejora
Framework: LangGraph
Caso de uso: Agente de código que aprende de sus errores

Reflexion permite que el agente evaluate su propio output, identifique errores
y se corrija iterativamente.

Instalación:
pip install langgraph langchain langchain-openai
"""

import os
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
EVALUATOR = ChatOpenAI(model="gpt-4o", temperature=0)


class ReflexionState(TypedDict):
    """Estado del agente Reflexion"""
    task: str
    code_attempt: str
    reflection: str
    iteration: int
    max_iterations: int
    is_correct: bool
    error_history: List[str]


def generate_code(state: ReflexionState) -> ReflexionState:
    """Paso 1: Generar código basándose en la tarea y reflexiones previas"""
    iteration = state["iteration"]
    
    print(f"\n{'='*70}")
    print(f"ITERACIÓN {iteration + 1}")
    print('='*70)
    print(f"\n💻 Generando código...")
    
    # Context de intentos previos
    context = ""
    if state["code_attempt"]:
        context = f"""
Intento previo:
```python
{state["code_attempt"]}
```

Reflexión sobre el error:
{state["reflection"]}

Historial de errores:
{chr(10).join(f'- {err}' for err in state["error_history"])}

APRENDE de estos errores y NO los repitas."""
    
    prompt = f"""Tarea: {state['task']}
{context}

Escribe código Python CORRECTO y COMPLETO que resuelva la tarea.
Incluye:
- Imports necesarios
- Funciones con docstrings
- Type hints
- Manejo de edge cases
- Código limpio y pythonico

Responde SOLO con el código, sin explicaciones adicionales."""
    
    response = LLM.invoke(prompt)
    code = response.content.strip()
    
    # Limpiar código (quitar markdown si existe)
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    print(f"\n✅ Código generado:")
    print(f"```python\n{code}\n```")
    
    return {**state, "code_attempt": code}


def evaluate_code(state: ReflexionState) -> ReflexionState:
    """Paso 2: Evaluar el código de forma crítica"""
    print(f"\n⚖️ Evaluando código...")
    
    eval_prompt = f"""Tarea original: {state['task']}

Código a evaluar:
```python
{state['code_attempt']}
```

Evalúa críticamente este código considerando:
1. ¿Resuelve correctamente la tarea?
2. ¿Tiene bugs o errores lógicos?
3. ¿Maneja edge cases?
4. ¿Sigue buenas prácticas (PEP 8, type hints, docstrings)?
5. ¿Es eficiente?

Responde en formato:
CORRECTO: SI/NO
PROBLEMAS: [lista de problemas encontrados, o "ninguno"]
SUGERENCIAS: [mejoras específicas]"""
    
    response = EVALUATOR.invoke(eval_prompt)
    evaluation = response.content
    
    print(f"\n📝 Evaluación:")
    print(evaluation)
    
    # Determinar si es correcto
    is_correct = "CORRECTO: SI" in evaluation or "CORRECTO:SI" in evaluation.replace(" ", "")
    
    return {**state, "reflection": evaluation, "is_correct": is_correct}


def reflect_and_learn(state: ReflexionState) -> ReflexionState:
    """Paso 3: Reflexionar sobre errores y extraer lecciones"""
    print(f"\n🤔 Reflexionando sobre errores...")
    
    if state["is_correct"]:
        print("✅ No hay errores que reflexionar, código es correcto!")
        return state
    
    reflection_prompt = f"""Analiza esta evaluación y extrae una lección concisa:

{state['reflection']}

¿Cuál fue el ERROR PRINCIPAL cometido? (1 línea, específica)"""
    
    response = LLM.invoke(reflection_prompt)
    main_error = response.content.strip()
    
    print(f"   Lección aprendida: {main_error}")
    
    # Agregar al historial
    error_history = state["error_history"] + [main_error]
    
    return {**state, "error_history": error_history}


def should_continue(state: ReflexionState) -> str:
    """Decisión: ¿Continuar iterando?"""
    if state["is_correct"]:
        print("\n🎉 ¡Código correcto encontrado!")
        return "end"
    
    if state["iteration"] + 1 >= state["max_iterations"]:
        print(f"\n⏱️ Máximo de iteraciones ({state['max_iterations']}) alcanzado")
        return "end"
    
    print(f"\n🔄 Continuando a la siguiente iteración...")
    return "generate"


def increment_iteration(state: ReflexionState) -> ReflexionState:
    """Incrementar contador de iteración"""
    return {**state, "iteration": state["iteration"] + 1}


def create_reflexion_agent():
    """Crear grafo de agente Reflexion"""
    workflow = StateGraph(ReflexionState)
    
    # Nodos
    workflow.add_node("generate", generate_code)
    workflow.add_node("evaluate", evaluate_code)
    workflow.add_node("reflect", reflect_and_learn)
    workflow.add_node("increment", increment_iteration)
    
    # Flujo
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", "reflect")
    workflow.add_edge("reflect", "increment")
    
    # Loop o fin
    workflow.add_conditional_edges(
        "increment",
        should_continue,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    return workflow.compile()


def main():
    """Demostración de Reflexion Agent"""
    print("=" * 70)
    print("Agente Reflexion - Auto-Mejora a través de Reflexión")
    print("=" * 70)
    
    # Tareas de prueba
    tasks = [
        """Escribe una función 'fibonacci(n)' que retorne el n-ésimo número de Fibonacci.
Debe manejar n=0, n=1 y n < 0 correctamente.
Usa recursión con memoization para eficiencia.""",
        
        # """Escribe una función 'is_palindrome(s)' que determine si una cadena es palíndromo.
        # Debe ignorar espacios, mayúsculas y caracteres especiales.
        # Ejemplo: "A man a plan a canal Panama" -> True"""
    ]
    
    agent = create_reflexion_agent()
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"TAREA {i}: {task[:80]}...")
        print('='*70)
        
        initial_state = ReflexionState(
            task=task,
            code_attempt="",
            reflection="",
            iteration=0,
            max_iterations=3,
            is_correct=False,
            error_history=[]
        )
        
        result = agent.invoke(initial_state)
        
        print(f"\n{'='*70}")
        print("RESULTADO FINAL")
        print('='*70)
        print(f"\nIteraciones usadas: {result['iteration'] + 1}")
        print(f"Código final:\n```python\n{result['code_attempt']}\n```")
        
        if result['is_correct']:
            print("\n✅ Estado: CORRECTO")
        else:
            print("\n⚠️ Estado: No se logró código perfecto en el límite de iteraciones")
            print(f"\nProblemas restantes:\n{result['reflection']}")
        
        print(f"\nErrores aprendidos durante el proceso:")
        for j, error in enumerate(result['error_history'], 1):
            print(f"  {j}. {error}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    main()
