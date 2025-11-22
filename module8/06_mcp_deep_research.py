"""
06_mcp_deep_research.py
=======================
Este script implementa un "Deep Researcher" potenciado por MCP (Model Context Protocol).
Utiliza CrewAI para orquestar agentes que consumen herramientas a través de un servidor MCP.

Inspirado en: "Building an MCP-powered Deep Researcher" (Daily Dose of Data Science).

Arquitectura:
1.  **Web Search Agent:** Usa una herramienta MCP (ej. Bright Data) para buscar URLs.
2.  **Specialist Agent:** Analiza el contenido profundo de esas URLs.
3.  **Response Agent:** Sintetiza todo en un reporte coherente.

Requisitos:
pip install crewai mcp
"""

import os
from crewai import Agent, Task, Crew, Process
# from mcp import ClientSession, StdioServerParameters # (Conceptual import)

# Nota: En un entorno real, conectarías con un servidor MCP así:
# server_params = StdioServerParameters(command="npx", args=["-y", "@brightdata/mcp-server"])
# session = ClientSession(server_params)

# Simulamos las herramientas MCP para este ejemplo educativo
class MockMCPTools:
    def search_web(self, query):
        return f"[MCP Tool] Resultados de búsqueda para: {query} (URLs: example.com/a, example.com/b)"
    
    def scrape_content(self, url):
        return f"[MCP Tool] Contenido extraído de {url}: 'Datos profundos sobre el tema...'"

mcp_tools = MockMCPTools()

# --- 1. Definir Agentes ---

# Agente 1: El Buscador (Web Search Agent)
# Su trabajo es encontrar las fuentes, no leerlas todas.
search_agent = Agent(
    role='Web Search Specialist',
    goal='Encontrar las URLs más relevantes para {topic}',
    backstory="Eres experto en encontrar agujas en un pajar. Usas herramientas MCP para búsquedas profundas.",
    tools=[mcp_tools.search_web], # En realidad, envolverías la herramienta MCP
    verbose=True
)

# Agente 2: El Especialista (Research Specialist)
# Su trabajo es leer y extraer insights de las fuentes encontradas.
research_agent = Agent(
    role='Deep Insight Analyst',
    goal='Extraer hechos clave y datos duros de las URLs proporcionadas',
    backstory="Lees papers, blogs y reportes técnicos. Ignoras el ruido y extraes la señal.",
    tools=[mcp_tools.scrape_content],
    verbose=True
)

# Agente 3: El Sintetizador (Response Agent)
# Su trabajo es escribir el reporte final.
writer_agent = Agent(
    role='Lead Report Writer',
    goal='Escribir un reporte ejecutivo basado en los insights extraídos',
    backstory="Transformas datos crudos en narrativa convincente y bien citada.",
    verbose=True
)

# --- 2. Definir Tareas ---

task_search = Task(
    description="Busca fuentes autorizadas sobre: {topic}. Retorna una lista de 5 URLs prometedoras.",
    agent=search_agent,
    expected_output="Lista de URLs."
)

task_analyze = Task(
    description="Toma las URLs del paso anterior, extrae el contenido y resume los puntos clave.",
    agent=research_agent,
    expected_output="Resumen detallado de insights."
)

task_write = Task(
    description="Escribe un reporte final en Markdown citando las fuentes.",
    agent=writer_agent,
    expected_output="Reporte final en Markdown."
)

# --- 3. "Stitch them together" (Orquestación) ---

deep_research_crew = Crew(
    agents=[search_agent, research_agent, writer_agent],
    tasks=[task_search, task_analyze, task_write],
    process=Process.sequential,
    verbose=2
)

def main():
    topic = "Impacto de MCP (Model Context Protocol) en Agentes de IA"
    print(f"🚀 Iniciando MCP Deep Researcher para: {topic}\n")
    
    result = deep_research_crew.kickoff(inputs={'topic': topic})
    
    print("\n\n########################")
    print("## REPORTE FINAL ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    main()
