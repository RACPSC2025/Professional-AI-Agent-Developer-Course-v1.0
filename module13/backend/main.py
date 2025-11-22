"""
backend/main.py
===============
API Principal del Capstone Project.
Expone el "Software House" vía FastAPI y SSE para streaming.
"""

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents import app as agent_graph

app = FastAPI(title="Autonomous Software House API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    requirement: str

@app.post("/create-software")
async def create_software(job: JobRequest):
    """
    Endpoint que inicia el proceso de creación de software.
    Retorna un stream de eventos con el progreso de los agentes.
    """
    async def event_generator():
        initial_state = {
            "requirement": job.requirement,
            "messages": [],
            "iteration": 0,
            "plan": "",
            "code": "",
            "review_comments": ""
        }
        
        # Ejecutar el grafo de LangGraph
        # Nota: En producción, esto debería correr en un background task (Celery/Redis)
        # Aquí lo hacemos directo para demostración de streaming.
        
        yield f"data: 🚀 Iniciando proyecto: {job.requirement}\n\n"
        await asyncio.sleep(0.5)
        
        try:
            async for event in agent_graph.astream(initial_state):
                for node_name, state_update in event.items():
                    # Detectar qué agente habló
                    if node_name == "pm":
                        yield f"data: 👔 PM: Plan completado.\n\n"
                    elif node_name == "coder":
                        yield f"data: 👨‍💻 Coder: Código escrito.\n\n"
                    elif node_name == "qa":
                        review = state_update.get("review_comments", "")
                        if "APROBADO" in review:
                            yield f"data: 🧐 QA: ✅ Código Aprobado.\n\n"
                            # Enviar el código final
                            final_code = state_update.get("code", "").replace("\n", "\\n")
                            yield f"data: CODE_BLOCK: {final_code}\n\n"
                        else:
                            yield f"data: 🧐 QA: ❌ Cambios solicitados. Reintentando...\n\n"
                    
                    await asyncio.sleep(0.5) # Simular latencia visual
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
