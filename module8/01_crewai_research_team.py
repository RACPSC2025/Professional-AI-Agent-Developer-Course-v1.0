"""
01_crewai_research_team.py
==========================
Ejemplo Enterprise de CrewAI: Patrón Secuencial.

Este script demuestra cómo orquestar un equipo de agentes para transformar
un tema abstracto en un artículo de blog pulido.

Conceptos Clave:
1.  **Agents:** Roles especializados con "Backstory" para dar personalidad y contexto.
2.  **Tasks:** Unidades de trabajo atómicas con "Expected Output" claro.
3.  **Process:** Ejecución secuencial (Waterfall).

Requisitos:
pip install crewai langchain_openai
"""

import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Configuración de Modelo (Puede ser GPT-4 o local con Ollama)
# os.environ["OPENAI_API_KEY"] = "sk-..." 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# --- 1. Definición de Agentes (The Team) ---

# Agente 1: El Investigador
# Su trabajo es recopilar datos, no escribir bonito.
researcher = Agent(
    role='Senior Research Analyst',
    goal='Descubrir desarrollos de vanguardia en {topic}',
    backstory="""Trabajas en un Think Tank de tecnología líder.
    Tu especialidad es encontrar tendencias antes que nadie.
    Eres analítico, frío y basado en datos.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
    # tools=[SearchTool()] # En prod, aquí iría una herramienta real
)

# Agente 2: El Escritor
# Su trabajo es hacer que los datos sean aburridos suenen emocionantes.
writer = Agent(
    role='Tech Content Strategist',
    goal='Crear contenido atractivo sobre {topic}',
    backstory="""Eres un escritor famoso en Medium y Substack.
    Sabes cómo simplificar temas complejos para una audiencia general.
    Tu tono es optimista pero profesional.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- 2. Definición de Tareas (The Work) ---

# Tarea 1: Investigación
task1 = Task(
    description="""Realiza un análisis exhaustivo sobre {topic}.
    Identifica los pros, contras y las tendencias clave del mercado.
    Tu informe final debe ser una lista de viñetas con datos duros.""",
    agent=researcher,
    expected_output="Informe de análisis de tendencias con 5 puntos clave."
)

# Tarea 2: Escritura
task2 = Task(
    description="""Usando el informe del Investigador, escribe un artículo de blog.
    1. Usa un título pegadizo.
    2. Escribe una introducción enganchadora.
    3. Desarrolla los puntos clave.
    4. Añade una conclusión reflexiva.""",
    agent=writer,
    expected_output="Artículo de blog de 500 palabras en formato Markdown."
)

# --- 3. Formación del Equipo (The Crew) ---

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential, # Ejecución paso a paso: Tarea 1 -> Tarea 2
    verbose=2 # Nivel de detalle en los logs
)

# --- 4. Ejecución ---

if __name__ == "__main__":
    print("🚀 Iniciando el Crew de Investigación...")
    topic = "El futuro de los Agentes de IA en 2025"
    
    result = crew.kickoff(inputs={'topic': topic})
    
    print("\n\n########################")
    print("## RESULTADO FINAL ##")
    print("########################\n")
    print(result)
