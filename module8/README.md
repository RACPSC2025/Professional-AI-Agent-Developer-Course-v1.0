# Módulo 8: Sistemas Multi-Agente (MAS)

![Module 8 Header](../images/module8_header.png)

![Level](https://img.shields.io/badge/Nivel-Avanzado-C3B1E1?style=for-the-badge&logo=crewai&logoColor=white)
![Time](https://img.shields.io/badge/Tiempo-5_Horas-A7C7E7?style=for-the-badge&labelColor=2D2D44)
![Stack](https://img.shields.io/badge/Stack-CrewAI_|_AutoGen_|_LangGraph-C3B1E1?style=for-the-badge)

## 🎯 Objetivos del Módulo
Un solo agente es un empleado. Múltiples agentes son una empresa. En este módulo, aprenderás a orquestar equipos de agentes especializados que colaboran para resolver problemas que ninguno podría resolver por sí solo.

## 📚 Conceptos Clave

### 1. Patrones de Colaboración
-   **Secuencial:** Agente A -> Agente B -> Agente C.
-   **Jerárquico (Manager/Worker):** Un agente "Jefe" desglosa la tarea y delega a agentes "Trabajadores".
    role='Escritor Tech',
    goal='Escribir artículos virales',
    backstory='Tienes un estilo enganchante...'
)

# Definir Tareas
tarea1 = Task(description='Investiga sobre AI Agents...', agent=investigador)
tarea2 = Task(description='Escribe un post sobre eso...', agent=escritor)

# Formar la Crew
crew = Crew(
    agents=[investigador, escritor],
    tasks=[tarea1, tarea2],
    verbose=2
)

result = crew.kickoff()
```

---

<div align="center">

**[⬅️ Módulo Anterior](../module7/README.md)** | **[🏠 Inicio](../README.md)** | **[Siguiente Módulo ➡️](../module9/README.md)**

</div>

