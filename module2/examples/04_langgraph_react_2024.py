"""
Module 2 Enhancement: LangGraph ReAct Agent (2024 API)
Framework: LangGraph 0.2+
Objective: Modern ReAct agent with streaming, error handling, and state management

This example demonstrates:
- Latest create_react_agent API (2024)
- Streaming responses
- Robust error handling
- State checkpointing
- Tool validation
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
import os

# ============================================================================
# 1. TOOL DEFINITIONS
# ============================================================================

@tool
def search_web(query: Annotated[str, "The search query"]) -> str:
    """Search the web for current information.
    
    Args:
        query: Search query string
        
    Returns:
        Search results as formatted string
    """
    # In production, integrate with real search API (Tavily, SerpAPI, etc.)
    return f"🔍 Search results for '{query}':\n- Latest AI frameworks comparison\n- Best practices for agent development\n- Production deployment guides"


@tool
def calculate(
    expression: Annotated[str, "Mathematical expression to evaluate"]
) -> str:
    """Safely evaluate mathematical expressions.
    
    Args:
        expression: Math expression (e.g., "2 + 2", "sqrt(16)")
        
    Returns:
        Calculation result
    """
    try:
        # Safe evaluation (in production, use safer alternatives like numexpr)
        import math
        allowed_names = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"✅ Result: {result}"
    except Exception as e:
        return f"❌ Error calculating '{expression}': {str(e)}"


@tool
def get_weather(
    city: Annotated[str, "City name"]
) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city
        
    Returns:
        Weather information
    """
    # In production, integrate with weather API
    return f"🌤️ Weather in {city}: Sunny, 22°C, Light breeze"


# ============================================================================
# 2. AGENT SETUP
# ============================================================================

def create_modern_agent():
    """Create a LangGraph ReAct agent with latest API."""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True  # Enable streaming
    )
    
    # Define tools
    tools = [search_web, calculate, get_weather]
    
    # Create checkpointer for state persistence
    memory = MemorySaver()
    
    # Create agent with new API (2024)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,  # Enables state persistence
        state_modifier="""You are a helpful AI assistant with access to tools.
        
        Guidelines:
        - Always think step-by-step before using tools
        - Validate tool inputs before calling
        - Provide clear, concise responses
        - If uncertain, ask for clarification
        """
    )
    
    return agent


# ============================================================================
# 3. EXECUTION EXAMPLES
# ============================================================================

def example_basic_query():
    """Example 1: Basic query with tool usage."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Query with Tools")
    print("="*70 + "\n")
    
    agent = create_modern_agent()
    
    # Configuration for this conversation thread
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    query = "What's the weather in Madrid and calculate 25 * 4?"
    
    print(f"User: {query}\n")
    
    # Invoke agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    # Print final response
    final_message = result["messages"][-1]
    print(f"Agent: {final_message.content}\n")


def example_streaming():
    """Example 2: Streaming responses."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Streaming Responses")
    print("="*70 + "\n")
    
    agent = create_modern_agent()
    config = {"configurable": {"thread_id": "conversation-2"}}
    
    query = "Search for information about LangGraph and explain what you find"
    
    print(f"User: {query}\n")
    print("Agent (streaming): ", end="", flush=True)
    
    # Stream responses
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="values"  # Stream intermediate values
    ):
        # Get the last message in the current state
        if "messages" in chunk and chunk["messages"]:
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                # Print only new content (simple approach)
                print(".", end="", flush=True)
    
    print("\n✅ Streaming complete\n")


def example_multi_turn():
    """Example 3: Multi-turn conversation with memory."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Multi-turn Conversation")
    print("="*70 + "\n")
    
    agent = create_modern_agent()
    config = {"configurable": {"thread_id": "conversation-3"}}
    
    # Turn 1
    query1 = "Calculate the square root of 144"
    print(f"User: {query1}")
    
    result1 = agent.invoke(
        {"messages": [HumanMessage(content=query1)]},
        config=config
    )
    print(f"Agent: {result1['messages'][-1].content}\n")
    
    # Turn 2 (references previous context)
    query2 = "Now multiply that result by 3"
    print(f"User: {query2}")
    
    result2 = agent.invoke(
        {"messages": [HumanMessage(content=query2)]},
        config=config
    )
    print(f"Agent: {result2['messages'][-1].content}\n")


def example_error_handling():
    """Example 4: Error handling and recovery."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Error Handling")
    print("="*70 + "\n")
    
    agent = create_modern_agent()
    config = {"configurable": {"thread_id": "conversation-4"}}
    
    # Query that will cause calculation error
    query = "Calculate the result of dividing by zero: 10 / 0"
    
    print(f"User: {query}\n")
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        print(f"Agent: {result['messages'][-1].content}\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n🤖 LangGraph ReAct Agent - 2024 API Examples")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'\n")
        return
    
    # Run examples
    example_basic_query()
    example_streaming()
    example_multi_turn()
    example_error_handling()
    
    print("\n✅ All examples completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   - create_react_agent (latest API)")
    print("   - Streaming responses")
    print("   - State persistence with MemorySaver")
    print("   - Multi-turn conversations")
    print("   - Error handling")
    print("   - Tool validation")


if __name__ == "__main__":
    main()
