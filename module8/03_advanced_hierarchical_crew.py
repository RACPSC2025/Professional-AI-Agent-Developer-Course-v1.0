"""
Módulo 8 - Ejemplo Avanzado: Agencia de Desarrollo de Software Jerárquica
Framework: CrewAI
Caso de uso: Equipo completo de desarrollo con manager, developers, QA y documentation

Sistema multi-agente jerárquico donde un Product Manager delega tareas a
especialistas (Backend Dev, Frontend Dev, QA Engineer, Tech Writer).

Instalación:
pip install crewai crewai-tools langchain-openai python-dotenv
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuración LLM
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def create_software_agency():
    """Crear agencia de desarrollo de software completa"""
    
    # AGENTES
    
    # 1. Product Manager (Manager jerárquico)
    product_manager = Agent(
        role='Product Manager',
        goal='Coordinar el equipo de desarrollo para entregar features de alta calidad',
        backstory="""Eres un Product Manager experimentado con 10 años en tech.
        Tu fortaleza es descomponer requisitos complejos en tareas claras y asignarlas
        al especialista correcto. Conoces las capacidades de cada miembro del equipo.""",
        verbose=True,
        allow_delegation=True,  # Puede delegar tareas
        llm=LLM
    )
    
    # 2. Backend Developer
    backend_dev = Agent(
        role='Backend Developer',
        goal='Diseñar e implementar APIs robustas y escalables',
        backstory="""Eres un backend developer senior especializado en Python/Node.js.
        Diseñas arquitecturas limpias, escribes código eficiente y piensas en
        escalabilidad, seguridad y performance. Favoreces REST APIs y arquitecturas
        basadas en microservicios.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM
    )
    
    # 3. Frontend Developer
    frontend_dev = Agent(
        role='Frontend Developer',
        goal='Crear interfaces de usuario intuitivas y responsivas',
        backstory="""Eres un frontend developer experto en React y diseño UX.
        Creas interfaces hermosas, accesibles y performantes. Piensas en mobile-first,
        accesibilidad (a11y) y mejores prácticas de CSS moderno.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM
    )
    
    # 4. QA Engineer
    qa_engineer = Agent(
        role='QA Engineer',
        goal='Asegurar calidad mediante testing exhaustivo',
        backstory="""Eres un QA Engineer meticuloso con ojo para el detalle.
        Diseñas planes de testing comprehensivos (unit, integration, e2e), identificas
        edge cases y automatizas pruebas. Tu misión es encontrar bugs ANTES de producción.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM
    )
    
    # 5. Technical Writer
    tech_writer = Agent(
        role='Technical Writer',
        goal='Crear documentación clara y completa',
        backstory="""Eres un Technical Writer que convierte complejidad técnica en
        documentación simple. Escribes READMEs impecables, API docs y user guides
        que developers y usuarios adoran.""",
        verbose=True,
        allow_delegation=False,
        llm=LLM
    )
    
    return product_manager, backend_dev, frontend_dev, qa_engineer, tech_writer


def create_feature_development_tasks(pm, backend, frontend, qa, writer, feature_spec: str):
    """Crear tareas para desarrollar una feature completa"""
    
    # Tarea 1: Product Manager planifica
    task_planning = Task(
        description=f"""Analiza este requisito y crea un plan de desarrollo detallado:

{feature_spec}

Descompón en:
1. Requisitos funcionales claros
2. Especificaciones de API (endpoints necesarios)
3. Requisitos de UI/UX
4. Criterios de aceptación
5. Consideraciones de testing

Formato: Plan estructurado y accionable.""",
        agent=pm,
        expected_output="Plan de desarrollo detallado con especificaciones claras"
    )
    
    # Tarea 2: Backend implementa API
    task_backend = Task(
        description="""Basándote en el plan del PM, diseña e implementa el backend:

1. Diseña la arquitectura de la API (endpoints, modelos de datos)
2. Define el schema de la base de datos
3. Escribe pseudocódigo o código de ejemplo para endpoints clave
4. Documenta decisiones de diseño (por qué esta arquitectura)
5. Identifica dependencias externas

Resultado: Especificación técnica completa del backend.""",
        agent=backend,
        expected_output="Diseño técnico completo de API con código de ejemplo",
        context=[task_planning]  # Depende del plan
    )
    
    # Tarea 3: Frontend implementa UI
    task_frontend = Task(
        description="""Basándote en el plan y el diseño de API, crea el frontend:

1. Diseña la arquitectura de componentes React
2. Define el flujo de datos (state management)
3. Crea mockups de las vistas principales (describe en texto)
4. Especifica las llamadas a API que necesitas
5. Lista consideraciones de UX y accesibilidad

Resultado: Especificación de frontend con estructura de componentes.""",
        agent=frontend,
        expected_output="Diseño de UI/UX con arquitectura de componentes React",
        context=[task_planning, task_backend]
    )
    
    # Tarea 4: QA crea plan de testing
    task_qa = Task(
        description="""Basándote en todas las especificaciones, crea un plan de QA:

1. Test cases para endpoints de API (casos normales y edge cases)
2. Test cases para UI (flujos de usuario, validaciones)
3. Casos de integración (frontend + backend)
4. Escenarios de performance y seguridad
5. Checklist de regresión

Resultado: Plan de testing comprehensivo.""",
        agent=qa,
        expected_output="Plan de QA completo con test cases detallados",
        context=[task_backend, task_frontend]
    )
    
    # Tarea 5: Technical Writer documenta todo
    task_documentation = Task(
        description="""Crea documentación completa del feature:

1. README con overview y quick start
2. Documentación de API (endpoints, params, responses)
3. Guía de usuario del frontend
4. Decisiones de arquitectura y por qué
5. Troubleshooting common issues

Formato: Markdown profesional y bien estructurado.
Audiencia: Developers que usarán/mantendrán esto.""",
        agent=writer,
        expected_output="Documentación técnica completa en Markdown",
        context=[task_backend, task_frontend, task_qa]
    )
    
    return [task_planning, task_backend, task_frontend, task_qa, task_documentation]


def main():
    """Ejecutar agencia de desarrollo completa"""
    
    print("=" * 80)
    print("🏢 Software Development Agency - Hierarchical Multi-Agent System")
    print("=" * 80)
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    # Crear equipo
    pm, backend, frontend, qa, writer = create_software_agency()
    
    # Feature a desarrollar
    feature_request = """
FEATURE REQUEST: Sistema de Notificaciones en Tiempo Real

Descripción:
Los usuarios deben recibir notificaciones en tiempo real cuando:
- Reciben un nuevo mensaje
- Alguien comenta en su post
- Su tarea asignada cambia de estado

Requisitos:
- Las notificaciones deben aparecer inmediatamente (no polling)
- Debe funcionar en web y móvil
- El usuario puede marcar notificaciones como leídas
- Debe haber un centro de notificaciones con historial
- Límite de 50 notificaciones históricas por usuario

Constraints:
- La solución debe escalar a 100,000 usuarios concurrentes
- Latencia máxima: 500ms desde evento hasta notificación
- Presupuesto limitado (soluciones open-source preferidas)
"""
    
    print(f"\n📋 FEATURE REQUEST:\n{feature_request}\n")
    
    # Crear tareas
    tasks = create_feature_development_tasks(
        pm, backend, frontend, qa, writer, 
        feature_request
    )
    
    # Formar el Crew (modo jerárquico con PM como manager)
    dev_crew = Crew(
        agents=[pm, backend, frontend, qa, writer],
        tasks=tasks,
        process=Process.hierarchical,  # Proceso jerárquico
        manager_llm=LLM,  # LLM para el manager
        verbose=2
    )
    
    print("\n🚀 Iniciando desarrollo del feature...\n")
    print("=" * 80)
    
    # Ejecutar (esto tomará varios minutos)
    result = dev_crew.kickoff()
    
    # Mostrar resultado final
    print("\n" + "=" * 80)
    print("✅ DESARROLLO COMPLETADO")
    print("=" * 80)
    print("\n📄 DOCUMENTACIÓN FINAL:\n")
    print(result)
    print("\n" + "=" * 80)
    
    print("""
💡 RESULTADO:
   ✅ Product Manager coordinó el equipo
   ✅ Backend Developer diseñó la arquitectura de API
   ✅ Frontend Developer creó la UI/UX
   ✅ QA Engineer preparó plan de testing
   ✅ Technical Writer documentó todo

Este es un ejemplo de cómo múltiples agentes especializados colaboran
jerárquicamente para entregar una feature completa de software.
""")


if __name__ == "__main__":
    main()
