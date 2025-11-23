# Módulo 9: Metacognición y Auto-Evolución (System 2 Thinking)

![Module 9 Banner](../images/module9_banner.png)

> "En Noviembre 2025, los agentes ya no solo responden. Se detienen, piensan sobre su propio pensamiento (Metacognición) y se corrigen antes de hablar."

## 🎯 Objetivos del Módulo

La mayoría de los LLMs operan en "System 1" (rápido, intuitivo, propenso a errores). En este módulo, aprenderás a forzar el "System 2" (lento, deliberado, lógico) usando técnicas avanzadas de 2025.

**Lo que vas a dominar:**
1.  🧠 **Metacognitive Prompting (MP):** La técnica de Nov 2025 para introspección profunda.
2.  🛡️ **Self-Correction (SCoRe):** Agentes que detectan sus propios errores sin feedback humano.
3.  🧬 **DSPy 2.5:** Optimización automática de prompts basada en métricas.

---

## 📚 Conceptos Clave (Nov 2025)

### 1. Metacognitive Prompting (MP)

A diferencia de "Chain of Thought" (CoT) que solo razona sobre el problema, **MP** razona sobre el *proceso* de resolver el problema.

**El Agente se pregunta:**
- "¿Entendí realmente la intención del usuario?"
- "¿Tengo suficiente información o estoy alucinando?"
- "¿Mi estrategia actual es la más eficiente?"

### 2. Intrinsic Error Detection

Investigaciones de finales de 2025 demuestran que los modelos grandes (GPT-5.1, Claude 4.5) tienen una "capacidad latente" para detectar sus propios errores si se les da el tiempo de cómputo para reflexionar *después* de generar un borrador, pero *antes* de mostrarlo.

---

## 🌍 High Impact Social/Professional Example (Nov 2025)

> **Proyecto: "SocratesAI" - Tutor Adaptativo con Metacognición**
>
> Este ejemplo implementa un tutor de matemáticas que no solo da respuestas, sino que evalúa su propia pedagogía en tiempo real para adaptarse al estudiante.

### El Problema
Los tutores de IA tradicionales explican siempre igual. Si el estudiante no entiende, repiten la misma explicación, frustrando al usuario.

### La Solución
Un agente con un bucle metacognitivo que analiza la confusión del estudiante y *se critica a sí mismo*: "Mi explicación fue muy técnica. Debo simplificar y usar una analogía."

```python
"""
Project: SocratesAI
Pattern: Metacognitive Reflection Loop
Framework: LangGraph / OpenAI GPT-5.1
"""
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class TutorState(TypedDict):
    history: List[str]
    last_explanation: str
    student_confusion_level: int # 0-10
    internal_monologue: str
    strategy: str

def assess_understanding(state: TutorState):
    # El modelo analiza la respuesta del estudiante
    # "No entiendo nada de integrales" -> Confusion: 9
    print("🤔 Assessing student state...")
    return {"student_confusion_level": 9}

def metacognitive_reflection(state: TutorState):
    # EL PASO CRÍTICO (System 2)
    if state["student_confusion_level"] > 5:
        reflection = """
        AUTO-CRÍTICA: Mi explicación anterior sobre 'área bajo la curva' fue demasiado abstracta.
        FALLO: Usé terminología de cálculo sin analogías.
        CORRECCIÓN: Cambiar estrategia a 'Analogía Física' (velocidad/tiempo).
        """
        print(f"🧠 METACOGNITION: {reflection}")
        return {"strategy": "analogy_physics", "internal_monologue": reflection}
    return {"strategy": "continue_curriculum"}

def generate_explanation(state: TutorState):
    if state["strategy"] == "analogy_physics":
        response = "Imagina que vas en un coche. El velocímetro dice 100 km/h..."
    else:
        response = "La integral se define como el límite de la suma de Riemann..."
    
    print(f"👨‍🏫 Tutor: {response}")
    return {"last_explanation": response}

# Construcción del Grafo
workflow = StateGraph(TutorState)
workflow.add_node("assess", assess_understanding)
workflow.add_node("reflect", metacognitive_reflection)
workflow.add_node("teach", generate_explanation)

workflow.set_entry_point("assess")
workflow.add_edge("assess", "reflect")
workflow.add_edge("reflect", "teach")
workflow.add_edge("teach", END)

app = workflow.compile()
```

**Impacto Social:**
- **Educación Personalizada**: Democratiza el acceso a tutoría de alta calidad que se adapta al ritmo de aprendizaje de cada niño.
- **Reducción de Frustración**: Evita el abandono escolar por "no entender".

---

## 🛠️ Proyectos Prácticos

### 🧠 Proyecto 1: El Crítico de Código (Reflexion)
Un agente que escribe código, ejecuta los tests unitarios, lee los errores, y se auto-corrige en un bucle hasta que los tests pasan.

### 🛡️ Proyecto 2: Self-RAG Validator
Un sistema RAG que genera 3 respuestas candidatas y usa un LLM-Judge para evaluar cuál tiene mejor soporte documental antes de responder al usuario.

### 🧬 Proyecto 3: Prompt Optimizer (DSPy)
Un script que toma tu prompt inicial "malo" y usa un dataset de ejemplos para reescribirlo y optimizarlo automáticamente usando DSPy 2.5.

---

## 🚀 Próximos Pasos

➡️ **[Módulo 10: Agentes Full Stack](../module10/README.md)**

<div align="center">

**[⬅️ Módulo Anterior](../module8/README.md)** | **[🏠 Inicio](../README.md)**

</div>

---

**Última actualización:** Noviembre 2025
**Stack:** LangGraph, DSPy 2.5
**Conceptos:** Metacognitive Prompting, System 2 Thinking
