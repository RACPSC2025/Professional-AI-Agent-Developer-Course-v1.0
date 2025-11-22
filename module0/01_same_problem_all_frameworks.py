"""
Módulo 0 - El Mismo Problema en 4 Frameworks
Caso de uso: Research Assistant que busca info y sintetiza

Compara la implementación del MISMO agente en:
1. LangChain
2. LangGraph  
3. CrewAI
4. AutoGen

Instalación:
pip install langchain langchain-openai langgraph crewai pyautogen python-dotenv
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("COMPARACIÓN PRÁCTICA DE FRAMEWORKS")
print("="*80)
print("\nImplementación del mismo Research Assistant en 4 frameworks diferentes\n")

# =============================================================================
# IMPLEMENTACIÓN 1: LANGCHAIN (Más simple, general purpose)
# =============================================================================

print("\n" + "="*80)
print("1. LANGCHAIN - General Purpose Framework")
print("="*80)

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def search_web_mock(query: str) -> str:
    """Simulación de búsqueda web"""
    return f"Search results for '{query}': AI agents are autonomous software..."

def summarize_mock(text: str) -> str:
    """Simulación de summarization"""
    return f"Summary: {text[:100]}..."

# Setup
llm_langchain = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

tools_langchain = [
    Tool(name="search", func=search_web_mock, description="Search the web"),
    Tool(name="summarize", func=summarize_mock, description="Summarize text")
]

prompt_langchain = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Search and synthesize information."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_langchain = create_openai_functions_agent(llm_langchain, tools_langchain, prompt_langchain)
executor_langchain = AgentExecutor(agent=agent_langchain, tools=tools_langchain, verbose=False)

# Ejecutar
result_langchain = executor_langchain.invoke({"input": "Research: what are AI agents?"})

print(f"\n✅ LangChain Result:")
print(f"   {result_langchain['output'][:150]}...")
print(f"\n📊 Lines of Code: ~25")
print(f"   Complexity: ⭐⭐ (Medium)")
print(f"   Setup Time: ~5 minutes")

# =============================================================================
# IMPLEMENTACIÓN 2: LANGGRAPH (Control explícito, state machine)
# =============================================================================

print("\n" + "="*80)
print("2. LANGGRAPH - Explicit State Machine")
print("="*80)

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    question: str
    search_results: Annotated[list, operator.add]
    summary: str

def search_node(state: ResearchState):
    results = [search_web_mock(state["question"])]
    return {"search_results": results}

def summarize_node(state: ResearchState):
    combined = "\n".join(state["search_results"])
    summary = llm_langchain.invoke(f"Summarize: {combined}").content
    return {"summary": summary}

# Build graph
workflow = StateGraph(ResearchState)
workflow.add_node("search", search_node)
workflow.add_node("summarize", summarize_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", END)

app_langgraph = workflow.compile()

# Ejecutar
result_langgraph = app_langgraph.invoke({
    "question": "What are AI agents?",
    "search_results": [],
    "summary": ""
})

print(f"\n✅ LangGraph Result:")
print(f"   {result_langgraph['summary'][:150]}...")
print(f"\n📊 Lines of Code: ~35")
print(f"   Complexity: ⭐⭐⭐⭐ (High - pero más control)")
print(f"   Setup Time: ~10 minutes")

# =============================================================================
# IMPLEMENTACIÓN 3: CREWAI (Multi-agent simplificado)
# =============================================================================

print("\n" + "="*80)
print("3. CREWAI - Multi-Agent Simplified")
print("="*80)

from crewai import Agent, Task, Crew

# Agentes especializados
researcher_crewai = Agent(
    role='Researcher',
    goal='Search for information on AI agents',
    backstory='Expert at finding relevant information',
    verbose=False,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

writer_crewai = Agent(
    role='Writer',
    goal='Synthesize research into clear summary',
    backstory='Expert at creating concise summaries',
    verbose=False,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# Tareas
task_research = Task(
    description='Research: What are AI agents?',
    agent=researcher_crewai,
    expected_output="Research findings about AI agents"
)

task_write = Task(
    description='Create a concise summary of the research',
    agent=writer_crewai,
    expected_output="Summary of AI agents",
    context=[task_research]
)

# Crew
crew_crewai = Crew(
    agents=[researcher_crewai, writer_crewai],
    tasks=[task_research, task_write],
    verbose=False
)

# Ejecutar
result_crewai = crew_crewai.kickoff()

print(f"\n✅ CrewAI Result:")
print(f"   {str(result_crewai)[:150]}...")
print(f"\n📊 Lines of Code: ~20")
print(f"   Complexity: ⭐⭐ (Low - muy intuitivo)")
print(f"   Setup Time: ~3 minutes")

# =============================================================================
# IMPLEMENTACIÓN 4: AUTOGEN (Conversacional)
# =============================================================================

print("\n" + "="*80)
print("4. AUTOGEN - Conversational Multi-Agent")
print("="*80)

from autogen import AssistantAgent, UserProxyAgent

# Config
config_list = [{
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY")
}]

# Agentes
researcher_autogen = AssistantAgent(
    "researcher",
    system_message="You are a researcher. Search and gather information.",
    llm_config={"config_list": config_list, "temperature": 0.3}
)

user_proxy = UserProxyAgent(
    "user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False
)

# Ejecutar (conversación)
user_proxy.initiate_chat(
    researcher_autogen,
    message="Research and summarize: What are AI agents?"
)

# Extraer resultado
messages = user_proxy.chat_messages[researcher_autogen]
result_autogen = messages[-1]["content"]

print(f"\n✅ AutoGen Result:")
print(f"   {result_autogen[:150]}...")
print(f"\n📊 Lines of Code: ~15")
print(f"   Complexity: ⭐⭐⭐ (Medium - conversational paradigm)")
print(f"   Setup Time: ~5 minutes")

# =============================================================================
# COMPARACIÓN FINAL
# =============================================================================

print("\n" + "="*80)
print("COMPARACIÓN FINAL")
print("="*80)

comparison_table = """
| Framework    | LoC | Complexity    | Setup Time | Best For                    |
|--------------|-----|---------------|------------|-----------------------------|
| LangChain    | ~25 | ⭐⭐          | ~5 min     | General purpose, RAG        |
| LangGraph    | ~35 | ⭐⭐⭐⭐      | ~10 min    | Complex workflows, control  |
| CrewAI       | ~20 | ⭐⭐          | ~3 min     | Multi-agent teams           |
| AutoGen      | ~15 | ⭐⭐⭐        | ~5 min     | Conversational agents       |
"""

print(comparison_table)

print("\n💡 KEY TAKEAWAYS:")
print("   • CrewAI = Más simple para multi-agent")
print("   • LangGraph = Máximo control (pero más verboso)")
print("   • LangChain = Balance general")
print("   • AutoGen = Excelente para conversaciones")
print("\n✅ Todos resuelven el mismo problema, elige según tu necesidad!")
