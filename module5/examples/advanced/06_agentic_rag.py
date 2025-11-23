"""
Module 5 Enhancement: Agentic RAG - November 2025
Framework: LangChain + LangGraph 1.0
Objective: AI agent manages iterative reasoning and dynamic retrieval

This example demonstrates:
- Agentic RAG (major November 2025 trend)
- Iterative reasoning and retrieval
- Dynamic query reformulation
- Multi-step problem-solving
- Tool use for external information
- Adaptive strategy based on intermediate results
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, List
import operator
import os

# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class AgenticRAGState(TypedDict):
    """State for Agentic RAG workflow."""
    messages: Annotated[List, operator.add]
    question: str
    plan: List[str]
    current_step: int
    retrieved_contexts: Annotated[List[str], operator.add]
    intermediate_answers: Annotated[List[str], operator.add]
    final_answer: str


# ============================================================================
# 2. TOOLS FOR AGENTIC RAG
# ============================================================================

# Setup vectorstore (global for tool access)
_vectorstore = None

def get_vectorstore():
    """Lazy initialization of vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        documents = [
            "LangGraph 1.0 was released in October 2025 with durable execution, advanced streaming, and human-in-the-loop capabilities.",
            "OpenAI GPT-5.1 was released on November 13, 2025, featuring enhanced steerability, faster responses, and improved agentic workflows.",
            "CrewAI v1.1.0 (October 21, 2025) introduced multi-provider LLM support, stricter typing, and is 5.76x faster than LangGraph.",
            "Agentic RAG emerged as a major trend in November 2025, enabling AI agents to manage iterative reasoning and dynamic retrieval.",
            "Multimodal RAG integrates visual and text information, preserving visual context for improved accuracy.",
            "Graph-RAG uses knowledge graphs for multi-hop reasoning and deterministic traversal of entity relationships.",
            "The global RAG market is projected to reach $1.85 billion in 2025.",
            "Microsoft announced the Agent Framework at Ignite 2025, unifying Semantic Kernel and AutoGen into a single SDK.",
            "LangChain secured $125M Series B funding in October 2025, achieving a $1.25B valuation.",
            "GPT-5-Codex-Max was introduced on November 19, 2025, for long-running, project-scale coding work."
        ]
        embeddings = OpenAIEmbeddings()
        _vectorstore = Chroma.from_texts(documents, embeddings)
    return _vectorstore


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information.
    
    Args:
        query: Search query to find relevant documents
        
    Returns:
        Relevant information from the knowledge base
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant information found."
    
    results = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"Found {len(docs)} relevant documents:\n{results}"


@tool
def reformulate_query(original_query: str, context: str) -> str:
    """Reformulate a query based on intermediate findings.
    
    Args:
        original_query: The original question
        context: Context from previous retrieval
        
    Returns:
        Reformulated query for better retrieval
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Based on the original query and the context found so far, 
    reformulate the query to find missing information.
    
    Original Query: {original_query}
    
    Context Found: {context[:200]}...
    
    Reformulated Query:"""
    
    reformulated = llm.invoke(prompt).content
    return f"Reformulated query: {reformulated}"


@tool
def synthesize_information(question: str, contexts: str) -> str:
    """Synthesize information from multiple contexts into a coherent answer.
    
    Args:
        question: The original question
        contexts: All retrieved contexts combined
        
    Returns:
        Synthesized answer
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Synthesize the following information to answer the question.
    
    Question: {question}
    
    Information:
    {contexts}
    
    Provide a comprehensive, well-structured answer:"""
    
    answer = llm.invoke(prompt).content
    return answer


# ============================================================================
# 3. AGENTIC RAG NODES
# ============================================================================

def create_plan(state: AgenticRAGState) -> AgenticRAGState:
    """
    Agentic Step 1: Create a multi-step plan to answer the question.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Create a step-by-step plan to answer this question:
    
    Question: {state['question']}
    
    Break it down into 2-4 specific steps. Each step should be a clear action.
    Format as a numbered list."""
    
    plan_text = llm.invoke(prompt).content
    
    # Extract steps (simple parsing)
    steps = [line.strip() for line in plan_text.split('\n') if line.strip() and any(c.isdigit() for c in line[:3])]
    
    state["plan"] = steps
    state["current_step"] = 0
    
    state["messages"].append(AIMessage(content=f"📋 Plan created:\n{plan_text}"))
    
    return state


def agent_reasoning(state: AgenticRAGState) -> AgenticRAGState:
    """
    Agentic Step 2: Agent decides what to do next.
    Uses tools for retrieval and reasoning.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to LLM
    tools = [search_knowledge_base, reformulate_query, synthesize_information]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create context-aware prompt
    context = f"""
    Original Question: {state['question']}
    
    Plan: {', '.join(state['plan'])}
    Current Step: {state['current_step'] + 1}/{len(state['plan'])}
    
    Retrieved so far: {len(state.get('retrieved_contexts', []))} contexts
    Intermediate findings: {len(state.get('intermediate_answers', []))} answers
    
    Decide what to do next to make progress on the current step.
    """
    
    response = llm_with_tools.invoke([HumanMessage(content=context)])
    state["messages"].append(response)
    
    return state


def execute_tools(state: AgenticRAGState) -> AgenticRAGState:
    """
    Agentic Step 3: Execute tools chosen by the agent.
    """
    # Get the last AI message
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        # No tools to execute
        return state
    
    # Execute each tool call
    tools = [search_knowledge_base, reformulate_query, synthesize_information]
    tool_node = ToolNode(tools)
    
    # Execute tools
    result = tool_node.invoke({"messages": [last_message]})
    
    # Add tool results to messages
    state["messages"].extend(result["messages"])
    
    return state


def should_continue(state: AgenticRAGState) -> str:
    """
    Decision: Should we continue iterating or finalize?
    """
    # Check if we've completed all steps
    if state["current_step"] >= len(state["plan"]) - 1:
        return "finalize"
    
    # Check if we have enough information
    if len(state.get("retrieved_contexts", [])) >= 3:
        return "finalize"
    
    # Continue iterating
    state["current_step"] += 1
    return "continue"


def finalize_answer(state: AgenticRAGState) -> AgenticRAGState:
    """
    Agentic Step 4: Finalize the answer using all gathered information.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Compile all contexts
    all_contexts = "\n\n".join(state.get("retrieved_contexts", []))
    
    # Get conversation history
    conversation = "\n".join([
        f"{msg.type}: {msg.content[:100]}..." 
        for msg in state["messages"][-5:] 
        if hasattr(msg, 'content')
    ])
    
    prompt = f"""Based on the entire conversation and all retrieved information, 
    provide a comprehensive final answer.
    
    Original Question: {state['question']}
    
    Retrieved Information:
    {all_contexts}
    
    Recent Conversation:
    {conversation}
    
    Final Answer:"""
    
    final_answer = llm.invoke(prompt).content
    state["final_answer"] = final_answer
    
    return state


# ============================================================================
# 4. BUILD GRAPH
# ============================================================================

def build_agentic_rag_graph():
    """Build the Agentic RAG workflow graph."""
    workflow = StateGraph(AgenticRAGState)
    
    # Add nodes
    workflow.add_node("plan", create_plan)
    workflow.add_node("reason", agent_reasoning)
    workflow.add_node("execute", execute_tools)
    workflow.add_node("finalize", finalize_answer)
    
    # Add edges
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "reason")
    workflow.add_edge("reason", "execute")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "continue": "reason",  # Loop back for more reasoning
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Run Agentic RAG examples."""
    print("\n🤖 Agentic RAG - November 2025 Trend")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        return
    
    # Build the graph
    app = build_agentic_rag_graph()
    
    # Example questions requiring multi-step reasoning
    questions = [
        "Compare the release dates and key features of LangGraph 1.0 and CrewAI v1.1.0",
        "What are the major AI agent framework developments in late 2025?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}: {question}")
        print('='*70)
        
        # Run the graph
        result = app.invoke({
            "messages": [HumanMessage(content=question)],
            "question": question,
            "plan": [],
            "current_step": 0,
            "retrieved_contexts": [],
            "intermediate_answers": [],
            "final_answer": ""
        })
        
        # Display results
        print(f"\n📋 PLAN:")
        for j, step in enumerate(result['plan'], 1):
            print(f"   {j}. {step}")
        
        print(f"\n📝 FINAL ANSWER:")
        print(f"{result['final_answer']}\n")
        
        print(f"📊 Agentic Metrics:")
        print(f"   Steps in Plan: {len(result['plan'])}")
        print(f"   Iterations: {result['current_step'] + 1}")
        print(f"   Contexts Retrieved: {len(result['retrieved_contexts'])}")
        print(f"   Tool Calls: {sum(1 for msg in result['messages'] if hasattr(msg, 'tool_calls') and msg.tool_calls)}")
    
    print("\n✅ All examples completed!")
    print("\n💡 Agentic RAG Features Demonstrated:")
    print("   - Multi-step planning")
    print("   - Iterative reasoning")
    print("   - Dynamic tool selection")
    print("   - Query reformulation")
    print("   - Information synthesis")
    print("   - Adaptive strategy")


if __name__ == "__main__":
    main()
