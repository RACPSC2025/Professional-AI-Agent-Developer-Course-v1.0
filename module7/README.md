# Módulo 7: Planificación Avanzada y Razonamiento (LangGraph 1.0)

![Module 7 Banner](../images/module7_banner.png)

> "En Noviembre 2025, ya no solo ejecutamos agentes. Viajamos en el tiempo a través de sus pensamientos para corregir el futuro."

## 🎯 Objetivos del Módulo

Los agentes simples (ReAct) funcionan bien para tareas cortas. Pero para procesos críticos que duran días o semanas, necesitas **Durable Execution**. En este módulo dominarás las capacidades avanzadas de **LangGraph 1.0**:

- 🕰️ **Time Travel Debugging**: Rebobinar el estado del agente, corregir un error y bifurcar una nueva realidad.
- 💾 **Durable Execution**: Agentes que "duermen" y despiertan semanas después sin perder contexto.
- 🚦 **Human-in-the-Loop (HITL)**: Sistemas de aprobación robustos para acciones sensibles.
- 🌳 **Tree of Thoughts (ToT)**: Explorar múltiples futuros posibles antes de actuar.

---

## 📚 Conceptos Clave (Nov 2025)

### 1. Time Travel Debugging

LangGraph guarda cada paso del agente como un "Checkpoint". Esto te permite:
1.  **Replay**: Ver exactamente qué pensó el agente paso a paso.
2.  **Fork**: Volver al paso 3, cambiar el input del usuario, y ver un resultado diferente.
3.  **Fix**: Si el agente falló en producción, puedes bajar el estado, arreglar el código, y reanudar desde el error.

```mermaid
graph LR
    A[Inicio] --> B[Paso 1]
    B --> C[Paso 2 (Error)]
    C --> D[Fallo]
    
    B -.->|Time Travel & Fix| C_Fixed[Paso 2 (Corregido)]
    C_Fixed --> E[Éxito]
    
    style C fill:#E74C3C,color:#fff
    style C_Fixed fill:#2ECC71,color:#fff
```

### 2. Durable Execution (Persistencia)

A diferencia de un script de Python normal, un grafo de LangGraph con persistencia (Postgres/Sqlite) es inmortal. Si el servidor se reinicia, el agente continúa exactamente donde se quedó.

---

## 🌍 High Impact Social/Professional Example (Nov 2025)

> **Proyecto: "UrbanFlow" - Sistema de Planificación Urbana Adaptativa**
>
> Este ejemplo utiliza **Time Travel** y **Durable Execution** para gestionar cambios de infraestructura en una ciudad inteligente.

### El Problema
Aprobar un cambio de tráfico (ej. hacer peatonal una calle) toma meses y requiere aprobaciones de múltiples departamentos. Si algo sale mal, revertirlo es costoso.

### La Solución
Un agente de larga duración que gestiona el proceso de aprobación y usa **Time Travel** para simular el impacto antes de ejecutarlo.

```python
"""
Project: UrbanFlow
Framework: LangGraph 1.0 (Nov 2025)
Capabilities: Durable Execution, Time Travel, HITL
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
import operator

# 1. Definir Estado del Proyecto Urbano
class UrbanState(TypedDict):
    proposal_id: str
    impact_score: float
    approvals: Annotated[list, operator.add]
    status: str

# 2. Nodos del Proceso
def simulate_impact(state: UrbanState):
    print(f"🔄 Simulating traffic impact for {state['proposal_id']}...")
    # Logic to call traffic simulation model
    # Time Travel: We can fork here to test different parameters!
    return {"impact_score": 0.85, "status": "simulated"}

def department_approval(state: UrbanState):
    print("⚖️ Requesting Department Approval...")
    # Human-in-the-loop breakpoint happens here
    return {"status": "pending_approval"}

def execute_change(state: UrbanState):
    if len(state['approvals']) >= 2:
        print("🏗️ Executing infrastructure change...")
        return {"status": "executed"}
    else:
        print("⛔ Change rejected.")
        return {"status": "rejected"}

# 3. Construir Grafo con Persistencia
builder = StateGraph(UrbanState)
builder.add_node("simulate", simulate_impact)
builder.add_node("approve", department_approval)
builder.add_node("execute", execute_change)

builder.set_entry_point("simulate")
builder.add_edge("simulate", "approve")
builder.add_edge("approve", "execute")

# Checkpointer para Durable Execution & Time Travel
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory, interrupt_before=["execute"])

# 4. Ejecución (Simulación de Semanas en Segundos)
thread_config = {"configurable": {"thread_id": "proposal-101"}}

# Paso 1: Simulación
print("--- Day 1: Simulation ---")
graph.invoke({
    "proposal_id": "Pedestrian-MainSt", 
    "approvals": [], 
    "status": "new"
}, config=thread_config)

# ... Semanas después ...
print("\n--- Day 15: Approval Received ---")
# Resume execution with new state (Human Input)
graph.update_state(thread_config, {"approvals": ["TransportDept", "Mayor"]})
graph.resume(thread_config)
```

**Impacto Profesional:**
- **Simulación Segura**: Usamos *Time Travel* para probar "¿Qué pasa si el impacto es 0.9?" sin reiniciar todo el proceso.
- **Auditoría Total**: Cada decisión queda guardada en el historial del grafo.
- **Resiliencia**: El proceso sobrevive reinicios de servidor durante los meses de aprobación.

---

## 🛠️ Proyectos Prácticos

### 🟢 Nivel Básico: Agente Plan-and-Execute
Implementación clásica de separación de preocupaciones.

### 🟡 Nivel Intermedio: Tree of Thoughts (ToT)
Resolver problemas complejos explorando múltiples ramas de razonamiento.

### 🔴 Nivel Avanzado: Time Travel Debugger
Crear una herramienta CLI que permita "viajar" por el historial de ejecución de un agente y modificar sus decisiones pasadas.

---

## 🚀 Próximos Pasos

➡️ **[Módulo 8: Sistemas Multi-Agente](../module8/README.md)**

<div align="center">

**[⬅️ Módulo Anterior](../module6/README.md)** | **[🏠 Inicio](../README.md)**

</div>

---

**Última actualización:** Noviembre 2025
**Stack:** LangGraph 1.0, LangSmith
**Conceptos:** Time Travel, Durable Execution, HITL
