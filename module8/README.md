# Módulo 8: Sistemas Multi-Agente (MAS)

![Module 8 Header](../images/module8_header.png)

![Level](https://img.shields.io/badge/Nivel-Avanzado-C3B1E1?style=for-the-badge&logo=crewai&logoColor=white)
![Time](https://img.shields.io/badge/Tiempo-8_Horas-A7C7E7?style=for-the-badge&labelColor=2D2D44)
![Stack](https://img.shields.io/badge/Stack-CrewAI_|_AutoGen_|_LangGraph_|_Semantic_Kernel-C3B1E1?style=for-the-badge)

> *"El talento gana partidos, pero el trabajo en equipo y la inteligencia ganan campeonatos."* — Michael Jordan

---

## 🎯 Objetivos del Módulo

Un solo agente es limitado. Un **Sistema Multi-Agente (MAS)** es ilimitado. En este módulo, aprenderás a orquestar equipos de agentes especializados que colaboran para resolver problemas complejos, simulando departamentos enteros de una empresa.

Cubriremos el **"Big 5"** de la orquestación de agentes:

1.  🚣 **CrewAI:** Roles estructurados y procesos secuenciales.
2.  🤖 **Microsoft AutoGen:** Conversación y ejecución de código.
3.  🕸️ **LangGraph:** Control de estado y grafos complejos.
4.  🏢 **Microsoft Semantic Kernel:** Integración empresarial y Plugins.
5.  ☁️ **Google Vertex AI Agents:** Escalabilidad en la nube y modelos Gemini.

---

## 📚 Índice

1.  [Fundamentos de Sistemas Multi-Agente](#1-fundamentos-de-sistemas-multi-agente)
2.  [Patrones de Orquestación Visualizados](#2-patrones-de-orquestación-visualizados)
3.  [Comparativa Definitiva de Frameworks](#3-comparativa-definitiva-de-frameworks)
4.  [Proyectos Prácticos](#-proyectos-prácticos)

---

## 1. Fundamentos de Sistemas Multi-Agente

Un MAS se compone de múltiples agentes interactuando en un entorno compartido. La clave no es solo tener muchos agentes, sino cómo **colaboran**.

### Ventajas sobre un Agente Único
-   **Especialización:** Un agente "Coder" y un "Writer" son mejores que un agente "Generalista".
-   **Paralelismo:** Varios agentes pueden trabajar en sub-tareas simultáneamente.
-   **Robustez:** Si un agente falla, otro puede corregirlo (ej. Reviewer revisando código).

---

## 2. Patrones de Orquestación Visualizados

### A. El Orquestador (Jerárquico)
Un "Supervisor" central dirige a trabajadores especializados. Es ideal para tareas complejas que requieren coordinación centralizada.

![The Orchestrator](../../brain/d0c3bcfe-8fae-456d-b0ac-1bf2041655fe/module8_orchestrator_1763820953731.png)
*Figura 1: Un Supervisor dirigiendo a agentes especializados (Construcción, Análisis, Arte).*

### B. La Mesa Redonda (Colaborativo)
Agentes autónomos discuten y colaboran en igualdad de condiciones. Ideal para brainstorming y resolución creativa de problemas.

![The Roundtable](../../brain/d0c3bcfe-8fae-456d-b0ac-1bf2041655fe/module8_roundtable_1763820983046.png)
*Figura 2: Diferentes frameworks (CrewAI, AutoGen, SK) colaborando en una mesa redonda digital.*

---

## 3. Comparativa Definitiva de Frameworks

| Característica | 🚣 CrewAI | 🤖 AutoGen | 🕸️ LangGraph | 🏢 Semantic Kernel | ☁️ Vertex AI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Filosofía** | **Role-First** | **Conversation-First** | **State-First** | **Enterprise-First** | **Model-First** |
| **Mejor para...** | Procesos de negocio, Marketing | Coding, Data Science | Apps complejas, SaaS | Integración .NET/Python | Escala masiva, Google Cloud |
| **Curva** | ⭐ (Baja) | ⭐⭐ (Media) | ⭐⭐⭐ (Alta) | ⭐⭐ (Media) | ⭐⭐ (Media) |
| **Control** | Alto en roles | Medio (Chat) | Total (Grafo) | Alto (Plugins) | Alto (Tools) |

---

## 🛠️ Proyectos Prácticos

### 🚣 Proyecto 1: El Equipo de Investigación (CrewAI)
**Archivo:** [`01_crewai_research_team.py`](01_crewai_research_team.py)
-   **Objetivo:** Crear un reporte completo sobre una tecnología emergente.
-   **Roles:** Lead Researcher, Senior Analyst, Tech Writer.

### 🤖 Proyecto 2: El Equipo de Desarrollo (AutoGen)
**Archivo:** [`02_autogen_coding_team.py`](02_autogen_coding_team.py)
-   **Objetivo:** Resolver un problema matemático escribiendo y ejecutando código Python.
-   **Roles:** UserProxy (Admin), Assistant (Coder).

### 🕸️ Proyecto 3: El Supervisor Corporativo (LangGraph)
**Archivo:** [`03_langgraph_supervisor.py`](03_langgraph_supervisor.py)
-   **Objetivo:** Sistema de soporte que enruta consultas al especialista adecuado.
-   **Roles:** Supervisor, Billing, Tech Support.

### 🏢 Proyecto 4: Asistente Empresarial (Semantic Kernel)
**Archivo:** [`04_semantic_kernel_agent.py`](04_semantic_kernel_agent.py)
-   **Objetivo:** Agente de productividad que gestiona agenda y correos.
-   **Concepto:** Uso de **Plugins** y **Kernel** para orquestación.

### ☁️ Proyecto 5: Agente de Viajes (Vertex AI)
**Archivo:** [`05_google_vertex_agent.py`](05_google_vertex_agent.py)
-   **Objetivo:** Planificador de viajes escalable.
-   **Concepto:** Estructura de **Tools** y razonamiento con Gemini.

---

## 🎓 Referencias

-   **CrewAI Docs:** [docs.crewai.com](https://docs.crewai.com/)
-   **AutoGen Docs:** [microsoft.github.io/autogen](https://microsoft.github.io/autogen/)
-   **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
-   **Semantic Kernel:** [learn.microsoft.com/semantic-kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
-   **Vertex AI Agents:** [cloud.google.com/vertex-ai/docs/agents](https://cloud.google.com/vertex-ai/docs/agents)

---

<div align="center">

**[⬅️ Módulo Anterior](../module7/README.md)** | **[🏠 Inicio](../README.md)** | **[Siguiente Módulo ➡️](../module9/README.md)**

</div>
