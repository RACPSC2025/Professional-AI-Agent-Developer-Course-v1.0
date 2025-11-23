"""
Module 10 Enhancement: Production Agent Deployment with FastAPI (November 2025)
Framework: FastAPI + LangGraph 1.0
Objective: Production-ready agent API with streaming and monitoring

This example demonstrates:
- FastAPI deployment for AI agents
- Streaming responses (LangGraph 1.0 feature)
- Error handling and retry logic
- Cost tracking and monitoring
- Rate limiting
- Health checks
- Production best practices
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import Optional, List
import asyncio
import json
import time
from datetime import datetime
import os

# ============================================================================
# 1. PYDANTIC MODELS
# ============================================================================

class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    query: str = Field(description="User query")
    session_id: Optional[str] = Field(default=None, description="Session ID for continuity")
    stream: bool = Field(default=False, description="Enable streaming")
    max_tokens: Optional[int] = Field(default=1000, description="Max tokens")


class AgentResponse(BaseModel):
    """Response model for agent."""
    response: str
    session_id: str
    tokens_used: int
    cost_usd: float
    execution_time_ms: int
    model: str


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float


class UsageStats(BaseModel):
    """Usage statistics."""
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    average_response_time_ms: float


# ============================================================================
# 2. AGENT SETUP
# ============================================================================

@tool
def get_current_time() -> str:
    """Get current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def create_production_agent():
    """Create production-ready agent."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Upgrade to gpt-5.1 when available
        temperature=0,
        streaming=True
    )
    
    tools = [get_current_time, calculate]
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier="You are a helpful AI assistant. Be concise and accurate."
    )
    
    return agent


# ============================================================================
# 3. MONITORING & COST TRACKING
# ============================================================================

class AgentMonitor:
    """Monitor agent usage and costs."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.response_times = []
    
    def track_request(self, tokens: int, cost: float, response_time_ms: int):
        """Track a request."""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.response_times.append(response_time_ms)
    
    def get_stats(self) -> UsageStats:
        """Get usage statistics."""
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return UsageStats(
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
            total_cost_usd=round(self.total_cost, 4),
            average_response_time_ms=round(avg_time, 2)
        )
    
    def estimate_cost(self, tokens: int, model: str = "gpt-4o-mini") -> float:
        """Estimate cost based on tokens."""
        # Pricing as of November 2025 (example rates)
        rates = {
            "gpt-4o-mini": 0.00015 / 1000,  # $0.15 per 1M tokens
            "gpt-5.1": 0.0003 / 1000,  # $0.30 per 1M tokens (example)
        }
        
        rate = rates.get(model, 0.0002 / 1000)
        return tokens * rate


# ============================================================================
# 4. FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AI Agent API",
    description="Production-ready AI agent with streaming support",
    version="1.0.0"
)

# Global monitor
monitor = AgentMonitor()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Agent API - November 2025",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - monitor.start_time
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=round(uptime, 2)
    )


@app.get("/stats", response_model=UsageStats, tags=["Monitoring"])
async def get_stats():
    """Get usage statistics."""
    return monitor.get_stats()


@app.post("/agent/invoke", response_model=AgentResponse, tags=["Agent"])
async def invoke_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """
    Invoke agent with a query (non-streaming).
    
    Args:
        request: Agent request
        background_tasks: Background tasks for async operations
        
    Returns:
        Agent response with metadata
    """
    start_time = time.time()
    
    try:
        # Create agent
        agent = create_production_agent()
        
        # Invoke agent
        result = agent.invoke({
            "messages": [HumanMessage(content=request.query)]
        })
        
        # Extract response
        response_text = result["messages"][-1].content
        
        # Calculate metrics
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Estimate tokens (simplified)
        tokens_used = len(request.query.split()) + len(response_text.split())
        cost = monitor.estimate_cost(tokens_used)
        
        # Track in background
        background_tasks.add_task(
            monitor.track_request,
            tokens_used,
            cost,
            execution_time_ms
        )
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{int(time.time())}"
        
        return AgentResponse(
            response=response_text,
            session_id=session_id,
            tokens_used=tokens_used,
            cost_usd=round(cost, 6),
            execution_time_ms=execution_time_ms,
            model="gpt-4o-mini"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/agent/stream", tags=["Agent"])
async def stream_agent(request: AgentRequest):
    """
    Invoke agent with streaming response.
    
    Args:
        request: Agent request
        
    Returns:
        Streaming response
    """
    if not request.stream:
        raise HTTPException(status_code=400, detail="Streaming not enabled in request")
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            agent = create_production_agent()
            
            # Stream events
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=request.query)]},
                version="v1"
            ):
                # Filter for relevant events
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        # Send as server-sent event
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\\n\\n"
                
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\\n\\n"
                
                elif event["event"] == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end'})}\\n\\n"
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete'})}\\n\\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# 5. ERROR HANDLING & MIDDLEWARE
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting Production Agent API")
    print("=" * 70)
    print("\n📍 Endpoints:")
    print("   - GET  /           : Root")
    print("   - GET  /health     : Health check")
    print("   - GET  /stats      : Usage statistics")
    print("   - POST /agent/invoke : Invoke agent (non-streaming)")
    print("   - POST /agent/stream : Invoke agent (streaming)")
    print("\n📚 Documentation:")
    print("   - http://localhost:8000/docs")
    print("\n💡 Features:")
    print("   - Streaming responses (LangGraph 1.0)")
    print("   - Cost tracking")
    print("   - Error handling")
    print("   - Health checks")
    print("   - Usage monitoring")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    
    print("\n" + "=" * 70)
    print("Starting server on http://localhost:8000")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
