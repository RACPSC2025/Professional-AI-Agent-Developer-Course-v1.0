"""
Ejemplo Básico: "Hello World" en 4 Frameworks
Módulo 2 - Comparación de Frameworks

Objetivo: Implementar la MISMA funcionalidad en 4 frameworks diferentes
para comparar sintaxis, verbosidad y developer experience.

Tarea: Generar un análisis estructurado de un framework de IA
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# MODELO DE DATOS COMPARTIDO (Pydantic)
# ============================================================================

class FrameworkAnalysis(BaseModel):
    """Análisis estructurado de un framework"""
    name: str = Field(description="Nombre del framework")
    best_for: str = Field(description="Para qué es mejor este framework")
    difficulty: int = Field(ge=1, le=5, description="Dificultad de aprendizaje (1-5)")
    key_features: list[str] = Field(description="3 características principales")


# ============================================================================
# 1. LANGCHAIN - The Swiss Army Knife
# ============================================================================

def example_langchain():
    """Implementación con LangChain + LCEL"""
    print("\n" + "="*80)
    print("1. LANGCHAIN - Usando LCEL (LangChain Expression Language)")
    print("="*80)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    
    # Setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=FrameworkAnalysis)
    
    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en frameworks de IA. Responde en JSON."),
        ("human", "Analiza el framework: {framework_name}\n\n{format_instructions}")
    ])
    
    # LCEL Chain (composición con | operator)
    chain = prompt | llm | parser
    
    # Ejecutar
    result = chain.invoke({
        "framework_name": "LangChain",
        "format_instructions": parser.get_format_instructions()
    })
    
    print(f"✅ Resultado LangChain:")
    print(f"   Framework: {result['name']}")
    print(f"   Mejor para: {result['best_for']}")
    print(f"   Dificultad: {result['difficulty']}/5")
    print(f"   Features: {', '.join(result['key_features'][:2])}")
    
    print(f"\n📊 Líneas de código: ~15")
    print(f"💭 Impresión: Modular pero requiere entender LCEL")


# ============================================================================
# 2. CREWAI - Role-Based Simplicity
# ============================================================================

def example_crewai():
    """Implementación con CrewAI"""
    print("\n" + "="*80)
    print("2. CREWAI - Role-Based Multi-Agent")
    print("="*80)
    
    from crewai import Agent, Task, Crew
    
    # Definir agente analista
    analyst = Agent(
        role='Framework Analyst',
        goal='Analizar frameworks de IA de forma objetiva',
        backstory='Experto con 10 años analizando herramientas de desarrollo',
        verbose=False,
        allow_delegation=False
    )
    
    # Definir tarea
    analysis_task = Task(
        description="""Analiza CrewAI y proporciona:
        - Nombre
        - Para qué es mejor
        - Dificultad de aprendizaje (1-5)
        - 3 características clave
        
        Formato: JSON con campos name, best_for, difficulty, key_features""",
        agent=analyst,
        expected_output="Análisis JSON estructurado"
    )
    
    # Crear crew
    crew = Crew(
        agents=[analyst],
        tasks=[analysis_task],
        verbose=False
    )
    
    # Ejecutar
    result = crew.kickoff()
    
    print(f"✅ Resultado CrewAI:")
    print(f"   {result}")
    
    print(f"\n📊 Líneas de código: ~25")
    print(f"💭 Impresión: Muy intuitivo para role-based, más verboso")


# ============================================================================
# 3. AUTOGEN - Conversational Agents
# ============================================================================

def example_autogen():
    """Implementación con AutoGen"""
    print("\n" + "="*80)
    print("3. AUTOGEN - Conversational Multi-Agent")
    print("="*80)
    
    from autogen import AssistantAgent, UserProxyAgent
    
    # Configurar LLM
    llm_config = {
        "model": "gpt-4o-mini",
        "temperature": 0
    }
    
    # Agente asistente
    assistant = AssistantAgent(
        name="analyst",
        llm_config=llm_config,
        system_message="""Eres un analista de frameworks de IA.
        Proporciona análisis en formato JSON con: name, best_for, difficulty (1-5), key_features (lista)."""
    )
    
    # User proxy (termina automáticamente)
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False
    )
    
    # Iniciar conversación
    user_proxy.initiate_chat(
        assistant,
        message="Analiza el framework AutoGen siguiendo tu formato JSON"
    )
    
    # Obtener último mensaje
    last_message = user_proxy.last_message()["content"]
    
    print(f"✅ Resultado AutoGen:")
    print(f"   {last_message[:200]}...")
    
    print(f"\n📊 Líneas de código: ~20")
    print(f"💭 Impresión: Excelente para conversaciones, setup más elaborado")


# ============================================================================
# 4. PYDANTIC AI - Type-Safe & Modern
# ============================================================================

def example_pydantic_ai():
    """Implementación con Pydantic AI"""
    print("\n" + "="*80)
    print("4. PYDANTIC AI - Type-Safe Agents")
    print("="*80)
    
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel
    
    # Crear agente con tipo de salida
    model = OpenAIModel("gpt-4o-mini")
    
    agent = Agent(
        model=model,
        result_type=FrameworkAnalysis,  # Type-safe!
        system_prompt="""Eres un experto en frameworks de IA.
        Proporciona análisis precisos y estructurados."""
    )
    
    # Ejecutar (result es type-safe)
    result = agent.run_sync("Analiza Pydantic AI")
    
    # result.data es FrameworkAnalysis (validado por Pydantic)
    print(f"✅ Resultado Pydantic AI:")
    print(f"   Framework: {result.data.name}")
    print(f"   Mejor para: {result.data.best_for}")
    print(f"   Dificultad: {result.data.difficulty}/5")
    print(f"   Features: {', '.join(result.data.key_features[:2])}")
    
    print(f"\n📊 Líneas de código: ~10")
    print(f"💭 Impresión: Más limpio, type-safe, pero muy nuevo")


# ============================================================================
# COMPARACIÓN FINAL
# ============================================================================

def print_comparison():
    """Imprime comparación final"""
    print("\n" + "="*80)
    print("📊 COMPARACIÓN FINAL")
    print("="*80)
    
    comparison = """
    | Framework     | Líneas | Verbosidad | Type Safety | Curva Aprendizaje | Best For              |
    |---------------|--------|------------|-------------|-------------------|-----------------------|
    | LangChain     | ~15    | Media      | ⭐⭐        | Alta              | Flexibilidad máxima   |
    | CrewAI        | ~25    | Alta       | ⭐⭐        | Media             | Multi-agente simple   |
    | AutoGen       | ~20    | Alta       | ⭐⭐        | Media             | Conversaciones        |
    | Pydantic AI   | ~10    | Baja       | ⭐⭐⭐⭐⭐  | Baja              | Structured outputs    |
    
    CONCLUSIONES:
    
    ✅ PYDANTIC AI: Gana en simplicidad y type safety
       - Ideal para equipos que usan mypy/pyright
       - Perfecto para structured data extraction
       - MUY nuevo (puede tener bugs)
    
    ✅ LANGCHAIN: Gana en flexibilidad y ecosistema
       - Más componentes disponibles
       - Mejor documentación
       - Mayor comunidad
    
    ✅ CREWAI: Gana en intuición para multi-agente
       - Muy fácil de conceptualizar (roles/tareas)
       - Menos flexible que LangChain
    
    ✅ AUTOGEN: Gana en conversaciones complejas
       - Excelente para multi-agent chat
       - Code execution nativo
       - Respaldo de Microsoft
    """
    
    print(comparison)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                    MÓDULO 2: COMPARACIÓN DE FRAMEWORKS                      ║
    ║                     Ejemplo Básico - "Hello World" x4                       ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    Implementaremos la MISMA funcionalidad en 4 frameworks diferentes:
    1. LangChain (LCEL)
    2. CrewAI (Role-based)
    3. AutoGen (Conversational)
    4. Pydantic AI (Type-safe)
    
    Objetivo: Comparar sintaxis, verbosidad y developer experience
    """)
    
    # Nota: Descomenta las funciones que quieras probar
    # Requiere tener instalado cada framework y API keys configuradas
    
    try:
        # example_langchain()
        # example_crewai()
        # example_autogen()
        # example_pydantic_ai()
        
        print_comparison()
        
    except ImportError as e:
        print(f"\n⚠️  Framework no instalado: {e}")
        print("\nPara instalar todos los frameworks:")
        print("pip install langchain-openai crewai autogen pydantic-ai chromadb")
        print_comparison()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de tener configurada OPENAI_API_KEY")
        print_comparison()
