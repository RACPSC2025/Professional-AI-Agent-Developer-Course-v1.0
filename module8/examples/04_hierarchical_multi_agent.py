"""
Module 8 Enhancement: Hierarchical Multi-Agent System (November 2025)
Framework: LangGraph 1.0
Objective: Supervisor pattern with specialized worker agents

This example demonstrates:
- Hierarchical multi-agent architecture (November 2025 pattern)
- Supervisor agent coordinating worker agents
- Specialized agents for different tasks
- State management across agent hierarchy
- Dynamic task routing
- Result synthesis
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List, Literal
import operator
import os

# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class HierarchicalState(TypedDict):
    """State for hierarchical multi-agent system."""
    messages: Annotated[List, operator.add]
    task: str
    subtasks: List[dict]
    current_worker: str
    research_results: str
    code_results: str
    writing_results: str
    final_output: str
    next_agent: str


# ============================================================================
# 2. WORKER AGENT TOOLS
# ============================================================================

@tool
def search_documentation(query: str) -> str:
    """Search technical documentation.
    
    Args:
        query: Search query
        
    Returns:
        Documentation results
    """
    # Simulated documentation search
    docs = {
        "langgraph": "LangGraph 1.0: Graph-based agent orchestration with durable execution, streaming, and HITL support.",
        "crewai": "CrewAI v1.1.0: Role-based multi-agent framework, 5.76x faster than LangGraph.",
        "openai": "OpenAI GPT-5.1: Released Nov 13, 2025. Enhanced steerability, faster responses.",
        "rag": "RAG 2025: Agentic RAG, Multimodal RAG, Graph-RAG, Self-RAG, Corrective RAG (CRAG)."
    }
    
    for key, value in docs.items():
        if key in query.lower():
            return f"📚 Found: {value}"
    
    return "📚 Documentation found (simulated)"


@tool
def analyze_code(code_snippet: str) -> str:
    """Analyze code quality and suggest improvements.
    
    Args:
        code_snippet: Code to analyze
        
    Returns:
        Analysis results
    """
    return f"""🔍 Code Analysis:
    - Lines: {len(code_snippet.split('\\n'))}
    - Quality: Good
    - Suggestions: Add error handling, type hints
    - Security: No issues detected
    """


@tool
def generate_code(description: str) -> str:
    """Generate code based on description.
    
    Args:
        description: What the code should do
        
    Returns:
        Generated code
    """
    return f'''```python
# Generated code for: {description}

def solution():
    """Implementation of {description}"""
    # TODO: Implement logic
    pass

if __name__ == "__main__":
    solution()
```'''


@tool
def format_document(content: str, style: str = "markdown") -> str:
    """Format document in specified style.
    
    Args:
        content: Content to format
        style: Output style (markdown, html, plain)
        
    Returns:
        Formatted document
    """
    return f"""# Formatted Document

{content}

---
*Formatted in {style} style*
"""


# ============================================================================
# 3. WORKER AGENTS
# ============================================================================

def create_research_agent():
    """Create specialized research agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_documentation]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content="""You are a Research Specialist.
        Your role: Find accurate, up-to-date technical information.
        Focus on: Latest frameworks, APIs, and best practices.
        Be thorough and cite sources.""")
    )
    
    return agent


def create_code_agent():
    """Create specialized coding agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [generate_code, analyze_code]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content="""You are a Senior Software Engineer.
        Your role: Write clean, production-ready code.
        Focus on: Best practices, error handling, type safety.
        Always include documentation.""")
    )
    
    return agent


def create_writing_agent():
    """Create specialized writing agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [format_document]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content="""You are a Technical Writer.
        Your role: Create clear, engaging documentation.
        Focus on: Clarity, structure, accessibility.
        Make complex topics easy to understand.""")
    )
    
    return agent


# ============================================================================
# 4. SUPERVISOR AGENT
# ============================================================================

def supervisor_node(state: HierarchicalState) -> HierarchicalState:
    """
    Supervisor agent: Coordinates worker agents and synthesizes results.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Determine what to do next
    if not state.get("subtasks"):
        # Initial planning
        prompt = f"""You are a Supervisor coordinating a team of specialist agents.

Task: {state['task']}

Available agents:
- research_agent: Finds technical information
- code_agent: Writes and analyzes code
- writing_agent: Creates documentation

Break down the task into 2-3 subtasks and assign each to an agent.
Respond in this format:
1. [agent_name]: subtask description
2. [agent_name]: subtask description
"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse subtasks (simplified)
        subtasks = []
        for line in response.content.split('\n'):
            if line.strip() and '[' in line:
                agent = line.split('[')[1].split(']')[0]
                task_desc = line.split(']: ')[1] if ']: ' in line else line
                subtasks.append({"agent": agent, "task": task_desc, "completed": False})
        
        state["subtasks"] = subtasks
        state["messages"].append(AIMessage(content=f"📋 Plan created: {len(subtasks)} subtasks"))
        
        # Start with first agent
        if subtasks:
            state["next_agent"] = subtasks[0]["agent"]
        
        return state
    
    # Check if all subtasks are complete
    incomplete = [st for st in state["subtasks"] if not st["completed"]]
    
    if not incomplete:
        # All done - synthesize final output
        prompt = f"""Synthesize the results from all agents into a final output.

Original Task: {state['task']}

Research Results: {state.get('research_results', 'N/A')}
Code Results: {state.get('code_results', 'N/A')}
Writing Results: {state.get('writing_results', 'N/A')}

Create a comprehensive final output."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["final_output"] = response.content
        state["next_agent"] = "FINISH"
        state["messages"].append(AIMessage(content="✅ All subtasks complete, synthesizing final output"))
        
        return state
    
    # Route to next incomplete subtask
    next_subtask = incomplete[0]
    state["next_agent"] = next_subtask["agent"]
    state["current_worker"] = next_subtask["agent"]
    state["messages"].append(AIMessage(content=f"➡️  Routing to {next_subtask['agent']}"))
    
    return state


def research_worker_node(state: HierarchicalState) -> HierarchicalState:
    """Research agent execution."""
    agent = create_research_agent()
    
    # Find research subtask
    subtask = next((st for st in state["subtasks"] if st["agent"] == "research_agent" and not st["completed"]), None)
    
    if subtask:
        print(f"\n🔬 Research Agent working on: {subtask['task']}")
        
        result = agent.invoke({
            "messages": [HumanMessage(content=subtask['task'])]
        })
        
        state["research_results"] = result["messages"][-1].content
        subtask["completed"] = True
        state["messages"].append(AIMessage(content=f"✅ Research complete"))
    
    return state


def code_worker_node(state: HierarchicalState) -> HierarchicalState:
    """Code agent execution."""
    agent = create_code_agent()
    
    # Find code subtask
    subtask = next((st for st in state["subtasks"] if st["agent"] == "code_agent" and not st["completed"]), None)
    
    if subtask:
        print(f"\n💻 Code Agent working on: {subtask['task']}")
        
        # Include research results as context if available
        context = f"Research context: {state.get('research_results', 'N/A')}\n\nTask: {subtask['task']}"
        
        result = agent.invoke({
            "messages": [HumanMessage(content=context)]
        })
        
        state["code_results"] = result["messages"][-1].content
        subtask["completed"] = True
        state["messages"].append(AIMessage(content=f"✅ Code complete"))
    
    return state


def writing_worker_node(state: HierarchicalState) -> HierarchicalState:
    """Writing agent execution."""
    agent = create_writing_agent()
    
    # Find writing subtask
    subtask = next((st for st in state["subtasks"] if st["agent"] == "writing_agent" and not st["completed"]), None)
    
    if subtask:
        print(f"\n✍️  Writing Agent working on: {subtask['task']}")
        
        # Include all previous results as context
        context = f"""Research: {state.get('research_results', 'N/A')}

Code: {state.get('code_results', 'N/A')}

Task: {subtask['task']}"""
        
        result = agent.invoke({
            "messages": [HumanMessage(content=context)]
        })
        
        state["writing_results"] = result["messages"][-1].content
        subtask["completed"] = True
        state["messages"].append(AIMessage(content=f"✅ Writing complete"))
    
    return state


def route_to_worker(state: HierarchicalState) -> str:
    """Route to appropriate worker agent."""
    next_agent = state.get("next_agent", "")
    
    if next_agent == "FINISH":
        return "finish"
    elif next_agent == "research_agent":
        return "research"
    elif next_agent == "code_agent":
        return "code"
    elif next_agent == "writing_agent":
        return "writing"
    else:
        return "supervisor"


# ============================================================================
# 5. BUILD HIERARCHICAL GRAPH
# ============================================================================

def build_hierarchical_graph():
    """Build hierarchical multi-agent workflow."""
    workflow = StateGraph(HierarchicalState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_worker_node)
    workflow.add_node("code", code_worker_node)
    workflow.add_node("writing", writing_worker_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "research": "research",
            "code": "code",
            "writing": "writing",
            "finish": END
        }
    )
    
    # All workers return to supervisor
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("code", "supervisor")
    workflow.add_edge("writing", "supervisor")
    
    return workflow.compile()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Run hierarchical multi-agent examples."""
    print("\n🏗️  Hierarchical Multi-Agent System - November 2025")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        return
    
    # Build the graph
    app = build_hierarchical_graph()
    
    # Example tasks
    tasks = [
        "Create a tutorial on implementing Self-RAG with LangGraph 1.0",
        "Build a production-ready multi-agent system with error handling"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}: {task}")
        print('='*70)
        
        # Run the hierarchical system
        result = app.invoke({
            "messages": [HumanMessage(content=task)],
            "task": task,
            "subtasks": [],
            "current_worker": "",
            "research_results": "",
            "code_results": "",
            "writing_results": "",
            "final_output": "",
            "next_agent": ""
        })
        
        # Display results
        print(f"\n📋 SUBTASKS EXECUTED:")
        for j, subtask in enumerate(result['subtasks'], 1):
            status = "✅" if subtask['completed'] else "⏳"
            print(f"   {j}. {status} [{subtask['agent']}] {subtask['task']}")
        
        print(f"\n📝 FINAL OUTPUT:")
        print("─" * 70)
        print(result['final_output'])
        print("─" * 70)
        
        print(f"\n📊 Execution Metrics:")
        print(f"   Subtasks: {len(result['subtasks'])}")
        print(f"   Agents Used: {len(set(st['agent'] for st in result['subtasks']))}")
        print(f"   Messages: {len(result['messages'])}")
    
    print("\n✅ All examples completed!")
    print("\n💡 Hierarchical Multi-Agent Features:")
    print("   - Supervisor pattern")
    print("   - Specialized worker agents")
    print("   - Dynamic task decomposition")
    print("   - State management across hierarchy")
    print("   - Result synthesis")
    print("   - Conditional routing")


if __name__ == "__main__":
    main()
