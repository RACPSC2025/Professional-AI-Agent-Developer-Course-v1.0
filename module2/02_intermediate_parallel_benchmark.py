"""
🟡 NIVEL INTERMEDIO: BENCHMARK PARALELO (THE ARENA)
---------------------------------------------------
En producción, la latencia importa.
Este script ejecuta una "carrera" entre dos modelos usando programación asíncrona (`asyncio`).
No espera a que uno termine para empezar el otro.

Métricas:
- Tiempo Total de Ejecución.
- Calidad de respuesta (Subjetiva/Visual).

Requisitos:
pip install langchain-openai asyncio
"""

import asyncio
import time
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Definir los contendientes
MODELS = [
    {"id": "gpt-4o-mini", "name": "⚡ GPT-4o Mini (The Speedster)"},
    {"id": "gpt-4o", "name": "🧠 GPT-4o (The Brain)"}
]

async def run_race(prompt: str):
    print(f"\n🏁 INICIANDO CARRERA CON PROMPT: '{prompt}'\n")
    
    tasks = []
    for model_info in MODELS:
        tasks.append(evaluate_model(model_info, prompt))
    
    # Ejecutar todos simultáneamente
    results = await asyncio.gather(*tasks)
    
    print("\n🏆 --- RESULTADOS FINALES ---")
    # Ordenar por tiempo
    results.sort(key=lambda x: x['time'])
    for i, res in enumerate(results):
        print(f"{i+1}. {res['name']} - ⏱️ {res['time']:.4f}s")
        print(f"   Respuesta: {res['content'][:100]}...\n")

async def evaluate_model(model_info, prompt):
    llm = ChatOpenAI(model=model_info["id"], temperature=0)
    
    start_time = time.perf_counter()
    print(f"   🚀 Lanzando {model_info['name']}...")
    
    try:
        # Llamada asíncrona real
        response = await llm.ainvoke(prompt)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        print(f"   ✅ {model_info['name']} terminó en {duration:.2f}s")
        
        return {
            "name": model_info["name"],
            "time": duration,
            "content": response.content
        }
    except Exception as e:
        return {
            "name": model_info["name"],
            "time": 999.99,
            "content": f"Error: {e}"
        }

if __name__ == "__main__":
    # El loop de eventos de asyncio
    prompt_text = "Explica la teoría de la relatividad general en un párrafo conciso."
    asyncio.run(run_race(prompt_text))
