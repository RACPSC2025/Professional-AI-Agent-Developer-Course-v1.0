"""
🟢 NIVEL BÁSICO: CREWAI - EQUIPO MULTI-AGENTE PARA CREACIÓN DE CONTENIDO
------------------------------------------------------------------------
Este ejemplo demuestra CrewAI para orquestar múltiples agentes especializados.
Caso de Uso: Agencia de contenido digital con investigador, escritor y editor.

Conceptos Clave:
- CrewAI: Framework para equipos de agentes colaborativos
- Roles especializados: Cada agente tiene un rol claro
- Tareas encadenadas: Output de un agente alimenta al siguiente
"""

import os
import sys
from dotenv import load_dotenv

# Nota: CrewAI requiere instalación: pip install crewai crewai-tools
# Para este ejemplo, proporcionamos estructura conceptual

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

print("""
="*70)
  👥 CREWAI - AGENCIA DE CONTENIDO DIGITAL
="*70)

IMPORTANTE: Este módulo requiere instalar CrewAI:
  pip install crewai crewai-tools

ESTRUCTURA DEL EJEMPLO (Conceptual):

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# 1. DEFINIR AGENTES
investigador = Agent(
    role='Investigador de Contenido',
    goal='Investigar temas trending y recopilar datos precisos',
    backstory='''Eres un investigador experto con experiencia en análisis de tendencias.
    Tu trabajo es encontrar información veraz y relevante.''',
    tools=[SerperDevTool()],  # Herramienta de búsqueda
    verbose=True
)

escritor = Agent(
    role='Escritor Creativo',
    goal='Crear contenido viral y engaging basado en investigación',
    backstory='''Eres un escritor con estilo único que sabe enganchar audiencias.
    Transformas datos secos en historias cautivadoras.''',
    verbose=True
)

editor = Agent(
    role='Editor Senior',
    goal='Refinar y optimizar contenido para SEO y readability',
    backstory='''Eres un editor meticuloso que asegura calidad profesional.
    Tu ojo crítico detecta errores y mejora coherencia.''',
    verbose=True
)

# 2. DEFINIR TAREAS
task_investigar = Task(
    description='''Investiga las últimas tendencias en Agentes de IA.
    Enfócate en: aplicaciones empresariales, frameworks populares, casos de éxito 2025.
    Fuentes: blogs tech, papers, GitHub trending.''',
    agent=investigador,
    expected_output='Informe de investigación con 5 puntos clave y fuentes'
)

task_escribir = Task(
    description='''Basándote en la investigación, escribe un artículo de blog de 800 palabras.
    Título: "5 Aplicaciones de Agentes de IA que Están Transformando Empresas en 2025"
    Tono: Profesional pero accesible. Incluye ejemplos concretos.''',
    agent=escritor,
    expected_output='Artículo completo en formato markdown',
    context=[task_investigar]  # Depende de la tarea anterior
)

task_editar = Task(
    description='''Edita el artículo para:
    - Corregir gramática y ortografía
    - Optimizar para SEO (keywords: "agentes IA", "AI agents", "automatización")
    - Mejorar estructura (headings, bullets, CTAs)
    - Verificar factual accuracy''',
    agent=editor,
    expected_output='Artículo final listo para publicación',
    context=[task_escribir]
)

# 3. FORMAR CREW (EQUIPO)
crew_contenido = Crew(
    agents=[investigador, escritor, editor],
    tasks=[task_investigar, task_escribir, task_editar],
    process=Process.sequential,  # Ejecutar en orden
    verbose=2
)

# 4. EJECUTAR
print("\\n🚀 Iniciando proceso de creación de contenido...\\n")
resultado = crew_contenido.kickoff()

print("\\n="*70)
print("  📄 CONTENIDO FINAL")
print("="*70)
print(resultado)
```

FLUJO DEL PROCESO:
==================
1. 👨‍🔬 INVESTIGADOR:
   - Busca tendencias en IA
   - Recopila datos de fuentes confiables
   - Genera informe estructurado

2. ✍️ ESCRITOR:
   - Lee informe del investigador
   - Crea narrativa engaging
   - Escribe artículo draft

3. 📝 EDITOR:
   - Revisa artículo del escritor
   - Optimiza SEO
   - Publica versión final

VENTAJAS DE CREWAI:
==================
✅ Cada agente tiene rol y expertise clara
✅ Tareas encadenadas (context sharing)
✅ Process types: Sequential, Hierarchical, Parallel
✅ Integración nativa con herramientas (Serper, Browserless, etc.)
✅ Ideal para workflows de producción de contenido

CASOS DE USO REALES:
====================
- Agencias de marketing (Content creation)
- Research firms (Due diligence automation)
- Software houses (Code review teams)
- News organizations (Article generation)

Para ejecutar este ejemplo:
1. Instalar: pip install crewai crewai-tools
2. Configurar OPENAI_API_KEY y SERPER_API_KEY (API de búsqueda)
3. Ejecutar el script completo
""")

print("\n💡 Este es un ejemplo conceptual. Para implementación real,")
print("   instala CrewAI y ejecuta el código comentado arriba.")
