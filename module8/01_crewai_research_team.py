"""
01_crewai_research_team.py
==========================
Este script demuestra cómo usar CrewAI para orquestar un equipo de agentes con roles definidos.
CrewAI brilla en procesos secuenciales donde cada agente tiene un "Backstory" y "Goal" claro.

Caso de Uso: Generar un reporte de investigación sobre una tecnología.

Requisitos:
pip install crewai langchain_openai duckduckgo-search
"""

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

# Herramienta de búsqueda
search_tool = DuckDuckGoSearchRun()

# 1. Definir Agentes (Roles)

# Agente 1: Investigador
researcher = Agent(
    role='Lead Research Analyst',
    goal='Descubrir desarrollos de vanguardia en {topic}',
    backstory="""Eres un analista senior en una gran empresa de tecnología.
    Tu trabajo es investigar las últimas noticias y tendencias.
    Tienes un ojo crítico para distinguir el hype de la realidad.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

# Agente 2: Escritor Técnico
writer = Agent(
    role='Tech Content Strategist',
    goal='Escribir contenido tech convincente sobre {topic}',
    backstory="""Eres un escritor técnico reconocido.
    Transformas conceptos complejos en narrativas fáciles de entender.
    Tu estilo es profesional pero accesible.""",
    verbose=True,
    allow_delegation=True # Puede pedir detalles extra al investigador si es necesario
)

# 2. Definir Tareas

# Tarea 1: Investigación
task1 = Task(
    description="""Realiza una investigación exhaustiva sobre {topic}.
    Identifica tendencias clave, jugadores principales y noticias recientes.
    Tu entregable debe ser un resumen detallado con puntos clave.""",
    agent=researcher,
    expected_output="Un informe detallado de 3 párrafos sobre las tendencias actuales."
)

# Tarea 2: Escritura
task2 = Task(
    description="""Usando el informe del investigador, escribe un artículo de blog sobre {topic}.
    El artículo debe tener una introducción enganchante, cuerpo informativo y conclusión.
    Debe estar formateado en Markdown.""",
    agent=writer,
    expected_output="Un artículo de blog en markdown de 500 palabras."
)

# 3. Definir la Crew (Equipo)
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2, # Nivel de log
    process=Process.sequential # Ejecución secuencial: Tarea 1 -> Tarea 2
)

def main():
    topic = "Agentic AI and Multi-Agent Systems"
    print(f"🚀 Iniciando CrewAI para investigar: {topic}\n")
    
    result = crew.kickoff(inputs={'topic': topic})
    
    print("\n\n########################")
    print("## RESULTADO FINAL ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    # Asegúrate de tener OPENAI_API_KEY en tu entorno
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    main()
