"""
Ejemplo Intermedio: Google ADK Tools Ecosystem
Framework: Google ADK
Nivel: 🟡 Intermedio
Objetivo: Explorar el ecosistema de tools de Google ADK

Conceptos:
- Built-in tools (Google Cloud)
- Gemini API tools
- Custom tools
- Tool composition
- MCP (Model Context Protocol) integration
"""

from google.adk.agents import LlmAgent
from google.adk.tools import tool
from google.adk.tools.gemini_api import search_google
from pydantic import BaseModel, Field
import os


# ============================================================================
# Parte 1: Built-in Tools - Google Search
# ============================================================================

def example_1_builtin_search():
    """
    Usar la herramienta de búsqueda integrada de Google
    """
    print("\n" + "="*80)
    print("EJEMPLO 1: Google Search Built-in Tool")
    print("="*80 + "\n")
    
    # Agente con Google Search tool
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="research_agent",
        description="Researches topics using Google Search",
        instruction="""You are a research assistant with access to real-time Google Search.
        Use the search tool to find current information.
        Cite your sources.""",
        tools=[search_google]  # Built-in tool de Google ADK
    )
    
    query = "What are the latest developments in AI agents as of 2024?"
    
    print(f"User: {query}")
    response = agent.run(query)
    print(f"\nAgent Response:\n{response.output_text}")
    
    print("\n💡 Esta búsqueda es REAL - usa Google Search API en tiempo real!")


# ============================================================================
# Parte 2: Custom Tools Avanzados
# ============================================================================

def example_2_advanced_custom_tools():
    """
    Tools personalizados con validación Pydantic
    """
    print("\n" + "="*80)
    print("EJEMPLO 2: Custom Tools con Pydantic Validation")
    print("="*80 + "\n")
    
    # Schema de entrada para el tool
    class AnalysisRequest(BaseModel):
        framework_name: str = Field(description="Name of the framework to analyze")
        aspect: str = Field(description="Aspect to analyze: performance, cost, complexity")
    
    # Schema de salida
    class AnalysisResult(BaseModel):
        framework: str
        aspect: str
        score: int = Field(ge=1, le=10)
        notes: str
    
    @tool
    def analyze_framework(request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze an AI framework on a specific aspect
        
        Args:
            request: Analysis parameters
            
        Returns:
            Detailed analysis with score
        """
        # Datos simulados - en producción, consultar DB
        analysis_db = {
            ("langchain", "performance"): (7, "Good for RAG, moderate overhead"),
            ("crewai", "complexity"): (6, "Medium learning curve, role-based"),
            ("google_adk", "cost"): (8, "Efficient with Gemini models"),
        }
        
        key = (request.framework_name.lower(), request.aspect.lower())
        score, notes = analysis_db.get(key, (5, "No data available"))
        
        return AnalysisResult(
            framework=request.framework_name,
            aspect=request.aspect,
            score=score,
            notes=notes
        )
    
    # Agente que usa el tool personalizado
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="framework_analyst",
        description="Analyzes AI frameworks",
        instruction="You analyze AI frameworks. Use the analyze_framework tool to get detailed scores.",
        tools=[analyze_framework]
    )
    
    query = "How complex is CrewAI framework?"
    
    print(f"User: {query}")
    response = agent.run(query)
    print(f"\nAgent: {response.output_text}")


# ============================================================================
# Parte 3: Tool Composition - Agente como Tool
# ============================================================================

def example_3_agent_as_tool():
    """
    Usar un agente como herramienta de otro agente (composición)
    """
    print("\n" + "="*80)
    print("EJEMPLO 3: Agent Composition (Agent as Tool)")
    print("="*80 + "\n")
    
    # Agente especializado en matemáticas
    math_agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="math_specialist",
        description="Solves complex math problems",
        instruction="You are a math expert. Solve problems step by step."
    )
    
    # Agente especializado en escritura
    writer_agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="writer_specialist",
        description="Writes clear explanations",
        instruction="You write clear, engaging explanations for technical topics."
    )
    
    # Agente supervisor que coordina
    supervisor = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="supervisor",
        description="Coordinates specialist agents",
        instruction="""You coordinate specialist agents.
        Delegate math problems to math_specialist.
        Delegate writing tasks to writer_specialist.
        Combine their outputs into a coherent response.""",
        tools=[math_agent, writer_agent]  # ¡Agentes como tools!
    )
    
    query = "Calculate 25 * 17 and explain how multiplication works"
    
    print(f"User: {query}")
    print("(Supervisor will delegate to math_specialist and writer_specialist)\n")
    
    response = supervisor.run(query)
    print(f"Supervisor Response:\n{response.output_text}")
    
    print("\n✨ Esto demuestra una arquitectura jerárquica!")


# ============================================================================
# Parte 4: Google Cloud Tools Integration
# ============================================================================

def example_4_google_cloud_tools():
    """
    Integración con herramientas de Google Cloud
    """
    print("\n" + "="*80)
    print("EJEMPLO 4: Google Cloud Tools (Conceptual)")
    print("="*80 + "\n")
    
    print("""
    Google ADK ofrece integraciones nativas con Google Cloud:
    
    📊 BigQuery Agent Analytics
    ────────────────────────────
    from google.adk.tools.google_cloud import BigQueryTool
    
    bq_tool = BigQueryTool(
        project_id="your-project",
        dataset_id="analytics"
    )
    
    agent = LlmAgent(
        model="gemini-2.0-flash-exp",
        tools=[bq_tool],
        instruction="Query BigQuery to answer data questions"
    )
    
    
    🗄️ Cloud SQL / AlloyDB (MCP Toolbox)
    ─────────────────────────────────────
    from google.adk.tools.google_cloud import MCPToolboxDatabase
    
    db_tool = MCPToolboxDatabase(
        connection_string="postgresql://..."
    )
    
    agent = LlmAgent(
        tools=[db_tool],
        instruction="Query database for customer information"
    )
    
    
    💻 Code Execution (Agent Engine)
    ─────────────────────────────────
    from google.adk.tools.google_cloud import CodeExecutionTool
    
    code_tool = CodeExecutionTool(
        runtime="python",
        sandbox=True
    )
    
    agent = LlmAgent(
        tools=[code_tool],
        instruction="Write and execute Python code safely"
    )
    """)
    
    print("\n💡 Estas herramientas requieren configuración de GCP")
    print("   Ver: https://google.github.io/adk-docs/tools/google-cloud-tools/")


# ============================================================================
# Parte 5: Comparación del Tool Ecosystem
# ============================================================================

def example_5_ecosystem_comparison():
    """
    Comparar el ecosistema de tools entre frameworks
    """
    print("\n" + "="*80)
    print("EJEMPLO 5: Tool Ecosystem Comparison")
    print("="*80 + "\n")
    
    comparison = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                     Tool Ecosystem Comparison                         ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║ Framework      │ Built-in Tools        │ Custom Tools  │ Composition ║
    ╠════════════════╪═══════════════════════╪═══════════════╪═════════════╣
    ║ Google ADK     │ ⭐⭐⭐⭐⭐            │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐⭐⭐  ║
    ║                │ - Google Search       │ @tool         │ Agent as    ║
    ║                │ - BigQuery           │ decorator     │ tool        ║
    ║                │ - Code Execution     │ Pydantic      │             ║
    ║                │ - Cloud tools        │ validation    │             ║
    ╠════════════════╪═══════════════════════╪═══════════════╪═════════════╣
    ║ LangChain      │ ⭐⭐⭐⭐             │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐⭐    ║
    ║                │ - Many integrations  │ @tool         │ Chains      ║
    ║                │ - 100+ connectors    │ decorator     │ within      ║
    ║                │                      │               │ chains      ║
    ╠════════════════╪═══════════════════════╪═══════════════╪═════════════╣
    ║ CrewAI         │ ⭐⭐⭐               │ ⭐⭐⭐⭐      │ ⭐⭐⭐       ║
    ║                │ - Basic toolkit      │ Custom func   │ Limited     ║
    ║                │                      │               │             ║
    ╠════════════════╪═══════════════════════╪═══════════════╪═════════════╣
    ║ AutoGen        │ ⭐⭐⭐⭐             │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐    ║
    ║                │ - Code execution     │ Function      │ Agent chat  ║
    ║                │ - Web browsing       │ calling       │             ║
    ╚════════════════╧═══════════════════════╧═══════════════╧═════════════╝
    
    🏆 VENTAJAS CLAVE DE GOOGLE ADK:
    
    1. Native Google Cloud Integration
       → BigQuery, Cloud SQL, Vertex AI sin configuración compleja
    
    2. Gemini API Tools
       → Google Search, Code Execution directamente integrados
    
    3. Agent as Tool Pattern
       → Composición jerárquica de agentes de forma nativa
    
    4. MCP Support
       → Model Context Protocol para herramientas estandarizadas
    
    5. Type Safety
       → Pydantic validation en inputs/outputs de tools
    """
    
    print(comparison)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            Google ADK - Advanced Tools Ecosystem                     ║
    ║                    Framework: Google ADK (2024)                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Este script explora el ecosistema de herramientas de Google ADK:
    1. Built-in tools (Google Search)
    2. Custom tools con Pydantic
    3. Agent composition (agents as tools)
    4. Google Cloud tools
    5. Ecosystem comparison
    """)
    
    # Verificar API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  ADVERTENCIA: GOOGLE_API_KEY no encontrada")
        print("   Algunos ejemplos no funcionarán sin la API key.")
    
    try:
        # example_1_builtin_search()  # Requiere API key real
        example_2_advanced_custom_tools()
        # example_3_agent_as_tool()
        # example_4_google_cloud_tools()
        example_5_ecosystem_comparison()
        
        print("\n" + "="*80)
        print("✅ Ejemplos completados!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
