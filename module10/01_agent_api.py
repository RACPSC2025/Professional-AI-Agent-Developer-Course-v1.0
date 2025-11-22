"""
01_agent_api.py
===============
Backend Enterprise-Grade para Agentes de IA usando FastAPI.

Este archivo demuestra prácticas de ingeniería de software profesional:
1.  **Settings Management:** Configuración vía variables de entorno (Pydantic Settings).
2.  **Dependency Injection:** Desacoplamiento de servicios.
3.  **Global Exception Handling:** Errores controlados y logueados.
4.  **Middleware:** CORS y Logging.
5.  **Async Streaming:** Respuesta en tiempo real.

Requisitos:
pip install fastapi uvicorn pydantic-settings langchain langchain-openai
"""

import time
import logging
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuración (Settings) ---
class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Agent API"
    VERSION: str = "2.0.0"
    OPENAI_API_KEY: Optional[str] = None # Se lee de env var
    DEBUG_MODE: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()

# --- 2. Logging ---
logging.basicConfig(level=logging.INFO if not settings.DEBUG_MODE else logging.DEBUG)
logger = logging.getLogger("agent_api")

# --- 3. Servicio del Agente (Lógica de Negocio) ---
class AgentService:
    """
    Servicio encapsulado que maneja la lógica del LLM.
    Se inyecta como dependencia en los endpoints.
    """
    def __init__(self):
        # En un caso real, aquí inicializarías tu grafo de LangGraph o Crew
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            streaming=True,
            api_key=settings.OPENAI_API_KEY or "mock-key" # Fallback para demo
        )
        self.prompt = ChatPromptTemplate.from_template(
            "Eres un consultor corporativo experto. Responde de forma concisa y profesional.\nPregunta: {question}"
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def stream_answer(self, question: str) -> AsyncGenerator[str, None]:
        """Genera tokens asíncronamente."""
        try:
            async for chunk in self.chain.astream({"question": question}):
                yield f"data: {chunk}\n\n"
                # Pequeño sleep para simular latencia de red si es local
                # await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error en generación LLM: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

# Dependencia para inyección
def get_agent_service():
    return AgentService()

# --- 4. Modelos de Datos (DTOs) ---
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="La pregunta del usuario")
    user_id: str = Field("guest", description="ID del usuario para tracking")
    temperature: float = Field(0.7, ge=0.0, le=2.0)

# --- 5. Definición de la App FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando servicios del Agente...")
    yield
    logger.info("🛑 Apagando servicios...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Middleware: CORS (Permitir acceso desde cualquier frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware: Logging de Tiempo de Respuesta
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Path: {request.url.path} | Time: {process_time:.4f}s")
    return response

# Manejador Global de Errores
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Error interno del servidor. Contacte a soporte."},
    )

# --- 6. Endpoints ---

@app.get("/health")
async def health_check():
    """Health check para Kubernetes/Docker."""
    return {"status": "healthy", "version": settings.VERSION}

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service)
):
    """
    Endpoint principal de chat con streaming.
    Usa inyección de dependencias para obtener el servicio.
    """
    logger.info(f"Chat request de {request.user_id}: {request.query[:50]}...")
    
    return StreamingResponse(
        service.stream_answer(request.query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    # Configuración para desarrollo
    uvicorn.run(
        "01_agent_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
