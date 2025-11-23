"""
Ejemplo Básico: Google Agent Development Kit (ADK)
Framework: Google ADK
Nivel: 🟢 Básico
Objetivo: Crear un agente simple con Google ADK y comparar con otros frameworks

Conceptos:
- LlmAgent: Agente powered por LLM
- model: Modelo de Google (Gemini)
- instruction: System prompt
- Tools: Funciones que el agente puede llamar
"""

from google.adk.agents import LlmAgent
from google.adk.tools import tool
import os

# ============================================================================
# Parte 1: Agente Más Simple Posible
# ============================================================================

def example_1_minimal_agent():
    """
    El agente más simple: solo modelo e instrucciones
    """
    print("\n" + "="*80)
    print("EJEMPLO 1: Agente Mínimo con Google ADK")
    print("="*80 + "\n")
    
    # Crear agente básico
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="geography_agent",
        description="Answers geography questions",
        instruction="You are a geography expert. Provide accurate information about countries and capitals."
    )
    
    # Ejecutar
    response = agent.run("What's the capital of France?")
    
    print(f"User: What's the capital of France?")
    print(f"Agent: {response.output_text}")
    print(f"\nTokens used: {response.usage}")


# ============================================================================
# Parte 2: Agente con Tools
# ============================================================================

def example_2_agent_with_tools():
    """
    Agente que puede llamar herramientas
    """
    print("\n" + "="*80)
    print("EJEMPLO 2: Agente con Herramientas")
    print("="*80 + "\n")
    
    # Definir herramientas usando decorador @tool
    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        # Simulado - en producción usar API real
        weather_data = {
            "Madrid": "Sunny, 25°C",
            "Paris": "Cloudy, 18°C",
            "London": "Rainy, 15°C"
        }
        return weather_data.get(city, f"No data for {city}")
    
    @tool
    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        try:
            # ADVERTENCIA: eval() solo para demo - usar sympyparse en producción
            result = eval(expression)
            return float(result)
        except:
            return "Invalid expression"
    
    # Crear agente con tools
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="assistant_agent",
        description="General purpose assistant with tools",
        instruction="""You are a helpful assistant with access to tools.
        Use the weather tool to get current weather.
        Use the calculator for math operations.
        Always explain your reasoning.""",
        tools=[get_weather, calculate]
    )
    
    # Query que requiere ambas herramientas
    query = "What's the weather in Madrid? Also, what's 25 * 4?"
    
    print(f"User: {query}")
    response = agent.run(query)
    print(f"Agent: {response.output_text}\n")
    
    # Mostrar tool calls
    if response.tool_calls:
        print("Tools usados:")
        for call in response.tool_calls:
            print(f"  - {call.name}({call.arguments})")


# ============================================================================
# Parte 3: Comparación con LangChain
# ============================================================================

def example_3_comparison():
    """
    Mismo agente en Google ADK vs LangChain - comparar sintaxis
    """
    print("\n" + "="*80)
    print("EJEMPLO 3: Comparación Google ADK vs LangChain")
    print("="*80 + "\n")
    
    # ---- GOOGLE ADK VERSION ----
    @tool
    def search_docs(query: str) -> str:
        """Search documentation"""
        return f"Found documentation for: {query}"
    
    adk_agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="docs_agent",
        instruction="Help users find documentation",
        tools=[search_docs]
    )
    
    print("📘 Google ADK Code:")
    print("""
    @tool
    def search_docs(query: str) -> str:
        return f"Found docs for: {query}"
    
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="docs_agent",
        instruction="Help users find documentation",
        tools=[search_docs]
    )
    """)
    
    # ---- LANGCHAIN EQUIVALENT ----
    print("\n📗 LangChain Equivalent:")
    print("""
    from langchain.tools import tool
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_openai_functions_agent
    
    @tool
    def search_docs(query: str) -> str:
        return f"Found docs for: {query}"
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_openai_functions_agent(
        llm=llm,
        tools=[search_docs],
        prompt=prompt_template  # Requiere prompt template
    )
    executor = AgentExecutor(agent=agent, tools=tools)
    """)
    
    print("\n✅ Ventajas de Google ADK:")
    print("  - Menos boilerplate (~5 líneas vs ~10)")
    print("  - No requiere prompt template separado")
    print("  - Integración nativa con Gemini")
    print("  - API más simple para casos básicos")
    
    print("\n✅ Ventajas de LangChain:")
    print("  - Más flexibility y opciones")
    print("  - Ecosystem más maduro (LangSmith, LangServe)")
    print("  - Más integraciones (100+)")
    print("  - Comunidad más grande")


# ============================================================================
# Parte 4: Structured Outputs con Pydantic
# ============================================================================

def example_4_structured_output():
    """
    Google ADK con structured outputs (Pydantic models)
    """
    print("\n" + "="*80)
    print("EJEMPLO 4: Structured Outputs")
    print("="*80 + "\n")
    
    from pydantic import BaseModel, Field
    
    # Definir schema de salida
    class FrameworkInfo(BaseModel):
        """Information about an AI framework"""
        name: str = Field(description="Framework name")
        best_for: str = Field(description="What it's best for")
        complexity: int = Field(ge=1, le=5, description="Complexity 1-5")
        year_released: int = Field(description="Year of release")
    
    # Agente que retorna structured output
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="framework_analyzer",
        instruction="Analyze AI frameworks and return structured information",
        output_schema=FrameworkInfo
    )
    
    response = agent.run("Analyze Google ADK framework")
    
    print("User: Analyze Google ADK framework")
    print(f"\nStructured Output:")
    print(f"  Name: {response.data.name}")
    print(f"  Best for: {response.data.best_for}")
    print(f"  Complexity: {response.data.complexity}/5")
    print(f"  Year: {response.data.year_released}")
    
    print("\n💡 Nota: Los datos son type-safe gracias a Pydantic!")


# ============================================================================
# Main - Ejecutar todos los ejemplos
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║         Google Agent Development Kit (ADK) - Ejemplos Básicos       ║
    ║                     Framework: Google ADK (2024)                     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Este script demuestra las características básicas de Google ADK:
    1. Agente mínimo
    2. Agente con herramientas
    3. Comparación con LangChain
    4. Structured outputs
    
    Prerequisitos:
    - pip install google-adk[all]
    - Set GOOGLE_API_KEY environment variable
    """)
    
    # Verificar API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY no encontrada en variables de entorno")
        print("   Configura tu API key: export GOOGLE_API_KEY='your-key'")
        exit(1)
    
    try:
        # Ejecutar ejemplos
        example_1_minimal_agent()
        # example_2_agent_with_tools()  # Descomentar para ejecutar
        # example_3_comparison()
        # example_4_structured_output()
        
        print("\n" + "="*80)
        print("✅ Ejemplos completados exitosamente!")
        print("="*80)
        
        print("\n📚 Próximos pasos:")
        print("  1. Ver 03_google_adk_tools.py para tools avanzados")
        print("  2. Ver 03_google_adk_comparison.py para comparativas detalladas")
        print("  3. Leer la documentación: https://google.github.io/adk-docs/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Asegúrate de tener google-adk instalado: pip install google-adk[all]")
