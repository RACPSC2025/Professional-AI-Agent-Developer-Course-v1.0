"""
Ejemplo Avanzado: Framework Comparison - Same Task, Different Frameworks
Framework: Google ADK vs LangChain vs CrewAI
Nivel: 🔴 Avanzado
Objetivo: Implementar la MISMA funcionalidad en 3 frameworks para comparar

Tarea: "Investigar y escribir un resumen sobre un topic técnico"
- Input: Topic (ej: "Quantum Computing")
- Process: Research → Summarize → Format
- Output: Resumen profesional de 200 palabras
"""

import os
import time
from typing import Dict, Any


# ============================================================================
# Versión 1: GOOGLE ADK
# ============================================================================

def implementation_google_adk(topic: str) -> Dict[str, Any]:
    """
    Implementación con Google ADK
    """
    print("\n" + "="*80)
    print("🔷 GOOGLE ADK IMPLEMENTATION")
    print("="*80 + "\n")
    
    from google.adk.agents import LlmAgent
    from google.adk.workflow_agents import SequentialAgent
    from google.adk.tools import tool
    
    start_time = time.time()
    
    # Tool para research (simulado)
    @tool
    def research_topic(topic: str) -> str:
        """Research a technical topic"""
        return f"Research data about {topic}: [Simulated research results...]"
    
    # Agente 1: Researcher
    researcher = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="researcher",
        description="Researches technical topics",
        instruction="Research the given topic thoroughly using available tools",
        tools=[research_topic]
    )
    
    # Agente 2: Writer
    writer = LlmAgent(
        model="gemini-2.0-flash-exp",
        name="writer",
        description="Writes professional summaries",
        instruction="Write a 200-word professional summary based on research. Be technical but clear."
    )
    
    # Workflow secuencial
    workflow = SequentialAgent(
        name="research_pipeline",
        agents=[researcher, writer]
    )
    
    # Ejecutar
    result = workflow.run(f"Research and summarize: {topic}")
    
    execution_time = time.time() - start_time
    
    print(f"✅ Resultado:\n{result.output_text}\n")
    print(f"⏱️  Tiempo: {execution_time:.2f}s")
    print(f"📊 Tokens: {result.usage}")
    
    return {
        "framework": "Google ADK",
        "lines_of_code": 35,  # Aproximado
        "execution_time": execution_time,
        "result": result.output_text,
        "complexity": "Low"
    }


# ============================================================================
# Versión 2: LANGCHAIN
# ============================================================================

def implementation_langchain(topic: str) -> Dict[str, Any]:
    """
    Implementación con LangChain
    """
    print("\n" + "="*80)
    print("🔗 LANGCHAIN IMPLEMENTATION")
    print("="*80 + "\n")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.tools import tool
    
    start_time = time.time()
    
    # Tool para research
    @tool
    def research_topic(topic: str) -> str:
        """Research a technical topic"""
        return f"Research data about {topic}: [Simulated research results...]"
    
    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Prompt para research
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Use tools to research topics."),
        ("user", "Research: {topic}")
    ])
    
    # Prompt para writing
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional writer. Write a 200-word summary."),
        ("user", "Research data: {research}\n\nWrite summary about: {topic}")
    ])
    
    # Chain 1: Research
    research_chain = (
        research_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Chain 2: Writing
    writing_chain = (
        {
            "research": lambda x: research_topic.invoke(x["topic"]),
            "topic": RunnablePassthrough()
        }
        | writing_prompt
        | llm
        | StrOutputParser()
    )
    
    # Ejecutar pipeline completo
    result = writing_chain.invoke({"topic": topic})
    
    execution_time = time.time() - start_time
    
    print(f"✅ Resultado:\n{result}\n")
    print(f"⏱️  Tiempo: {execution_time:.2f}s")
    
    return {
        "framework": "LangChain",
        "lines_of_code": 55,  # Aproximado
        "execution_time": execution_time,
        "result": result,
        "complexity": "Medium-High"
    }


# ============================================================================
# Versión 3: CREWAI
# ============================================================================

def implementation_crewai(topic: str) -> Dict[str, Any]:
    """
    Implementación con CrewAI
    """
    print("\n" + "="*80)
    print("👥 CREWAI IMPLEMENTATION")
    print("="*80 + "\n")
    
    from crewai import Agent, Task, Crew, Process
    
    start_time = time.time()
    
    # Agente 1: Researcher
    researcher = Agent(
        role='Senior Research Analyst',
        goal=f'Research {topic} comprehensively',
        backstory="""You are an expert research analyst with deep knowledge
        of technical topics. You find accurate, relevant information.""",
        verbose=False,
        allow_delegation=False
    )
    
    # Agente 2: Writer
    writer = Agent(
        role='Technical Writer',
        goal='Create clear, professional summaries',
        backstory="""You are a technical writer known for transforming
        complex research into accessible, engaging summaries.""",
        verbose=False,
        allow_delegation=False
    )
    
    # Tarea 1: Research
    research_task = Task(
        description=f"""Research {topic} thoroughly. Find key concepts,
        recent developments, and important applications.""",
        agent=researcher,
        expected_output="Detailed research notes"
    )
    
    # Tarea 2: Writing
    writing_task = Task(
        description=f"""Based on the research, write a 200-word professional
        summary about {topic}. Be technical but clear.""",
        agent=writer,
        expected_output="200-word professional summary"
    )
    
    # Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=False
    )
    
    # Ejecutar
    result = crew.kickoff()
    
    execution_time = time.time() - start_time
    
    print(f"✅ Resultado:\n{result}\n")
    print(f"⏱️  Tiempo: {execution_time:.2f}s")
    
    return {
        "framework": "CrewAI",
        "lines_of_code": 50,  # Aproximado
        "execution_time": execution_time,
        "result": str(result),
        "complexity": "Medium"
    }


# ============================================================================
# Análisis Comparativo
# ============================================================================

def compare_results(results: list[Dict[str, Any]]):
    """
    Comparar resultados de los 3 frameworks
    """
    print("\n" + "="*80)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*80 + "\n")
    
    # Tabla comparativa
    print("╔═══════════════════╦═══════════════╦══════════════╦═══════════════╗")
    print("║ Framework         ║ Lines of Code ║ Exec Time    ║ Complexity    ║")
    print("╠═══════════════════╬═══════════════╬══════════════╬═══════════════╣")
    
    for r in results:
        print(f"║ {r['framework']:17} ║ {r['lines_of_code']:13} ║ {r['execution_time']:11.2f}s ║ {r['complexity']:13} ║")
    
    print("╚═══════════════════╩═══════════════╩══════════════╩═══════════════╝")
    
    # Análisis detallado
    print("\n🔍 Detailed Analysis:\n")
    
    print("🏆 GOOGLE ADK:")
    print("  ✅ Least lines of code (~35 lines)")
    print("  ✅ Native Sequential workflow agent")
    print("  ✅ Integrated tools ecosystem")
    print("  ✅ Type-safe with Pydantic")
    print("  ⚠️  Newer framework (less community)")
    
    print("\n🏆 LANGCHAIN:")
    print("  ✅ Most mature ecosystem")
    print("  ✅ Extensive integrations (100+)")
    print("  ✅ LangSmith for observability")
    print("  ⚠️  More boilerplate (~55 lines)")
    print("  ⚠️  Complex for simple cases")
    
    print("\n🏆 CREWAI:")
    print("  ✅ Most intuitive for multi-agent")
    print("  ✅ Clear role-based abstractions")
    print("  ✅ Medium complexity (~50 lines)")
    print("  ⚠️  Less flexible than LangChain")
    print("  ⚠️  Smaller ecosystem")
    
    # Recomendaciones
    print("\n💡 WHEN TO USE EACH:\n")
    
    recommendations = """
    Use Google ADK when:
      → You need simple, clean code
      → You're in Google Cloud ecosystem
      → You want native tool composition
      → Type safety is important
      → Project is new (no legacy constraints)
    
    Use LangChain when:
      → You need maximum flexibility
      → You require many integrations
      → Observability is critical (LangSmith)
      → You have existing LangChain code
      → Complex RAG pipelines
    
    Use CrewAI when:
      → Multi-agent collaboration is primary feature
      → You want role-based abstractions
      → Team members are non-technical (clearer code)
      → Workflow is research/writing/validation
      → Medium complexity sweet spot
    """
    
    print(recommendations)
    
    # Winner por categoría
    print("\n🏅 CATEGORY WINNERS:\n")
    print("  📝 Code Simplicity: Google ADK (35 lines)")
    print("  🚀 Execution Speed: (Varies by API latency)")
    print("  🔧 Flexibility: LangChain (most options)")
    print("  👥 Multi-Agent UX: CrewAI (clearest abstractions)")
    print("  🔒 Type Safety: Google ADK (Pydantic native)")
    print("  🌍 Ecosystem: LangChain (100+ integrations)")
    print("  📚 Documentation: LangChain (most mature)")
    print("  ☁️  Cloud Native: Google ADK (Vertex AI)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║             Framework Comparison - Same Task Analysis                ║
    ║                  Google ADK vs LangChain vs CrewAI                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Task: "Research and summarize a technical topic"
    
    We'll implement the EXACT SAME functionality in 3 frameworks:
    1. Google ADK (Sequential workflow)
    2. LangChain (LCEL chains)
    3. CrewAI (Role-based crew)
    
    Then compare: lines of code, execution time, complexity
    
    ⚠️  NOTA: Requiere API keys configuradas:
       - GOOGLE_API_KEY (para Google ADK)
       - OPENAI_API_KEY (para LangChain y CrewAI)
    """)
    
    # Topic de ejemplo
    TOPIC = "Quantum Computing"
    
    # Almacenar resultados
    all_results = []
    
    # Ejecutar implementaciones
    try:
        print(f"\n🎯 Task: Research and summarize '{TOPIC}'\n")
        
        # Google ADK
        if os.getenv("GOOGLE_API_KEY"):
            result_adk = implementation_google_adk(TOPIC)
            all_results.append(result_adk)
        else:
            print("⚠️  Skipping Google ADK (no API key)")
        
        # LangChain
        if os.getenv("OPENAI_API_KEY"):
            result_langchain = implementation_langchain(TOPIC)
            all_results.append(result_langchain)
        else:
            print("⚠️  Skipping LangChain (no API key)")
        
        # CrewAI
        if os.getenv("OPENAI_API_KEY"):
            result_crewai = implementation_crewai(TOPIC)
            all_results.append(result_crewai)
        else:
            print("⚠️  Skipping CrewAI (no API key)")
        
        # Comparar
        if all_results:
            compare_results(all_results)
        else:
            print("\n❌ No frameworks executed (missing API keys)")
        
        print("\n" + "="*80)
        print("✅ Comparison completed!")
        print("="*80)
        
        print("\n📚 Next Steps:")
        print("  1. Run this script with your API keys")
        print("  2. Try different topics to see consistency")
        print("  3. Measure cost with different models")
        print("  4. Read Module 2 README for more comparisons")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
