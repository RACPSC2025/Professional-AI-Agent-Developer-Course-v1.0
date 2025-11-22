"""
03_agent_evaluation_pipeline.py
===============================
Pipeline de Evaluación Automática con Ragas.

Este script simula un proceso de CI/CD (Integración Continua) para tu Agente.
Evalúa la calidad de las respuestas usando métricas objetivas.

Métricas usadas:
1.  **Faithfulness:** ¿La respuesta es fiel al contexto recuperado?
2.  **Answer Relevance:** ¿La respuesta contesta la pregunta del usuario?

Requisitos:
pip install ragas datasets
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configuración
# os.environ["OPENAI_API_KEY"] = "sk-..."

# --- 1. Dataset de Prueba ("Golden Dataset") ---
# En producción, esto vendría de un CSV o JSON anotado por humanos.
data = {
    'question': [
        '¿Quién ganó la copa del mundo 2022?',
        '¿Cuál es la capital de Francia?',
        'Explica la teoría de la relatividad en 5 palabras.'
    ],
    'answer': [
        'Argentina ganó la copa del mundo.', # Respuesta del Agente
        'París es la capital.',
        'Energía igual a masa por velocidad.'
    ],
    'contexts': [
        ['Argentina derrotó a Francia en la final de Qatar 2022.'], # Contexto recuperado (RAG)
        ['París es la ciudad más poblada y capital de Francia.'],
        ['E=mc^2 es la fórmula famosa de Einstein.']
    ],
    'ground_truth': [
        'Argentina',
        'París',
        'E=mc^2 relaciona energía y masa.'
    ]
}

def run_evaluation():
    print("🧪 Iniciando Evaluación de Calidad del Agente...")
    
    # Convertir a formato HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    
    # Ejecutar Ragas
    # Ragas usa un LLM (GPT-4/3.5) como "Juez" para calificar al Agente.
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        embeddings=OpenAIEmbeddings()
    )
    
    print("\n📊 Reporte de Resultados:")
    print(results)
    
    # Validación tipo CI/CD
    df = results.to_pandas()
    avg_faithfulness = df["faithfulness"].mean()
    
    print(f"\nPromedio Faithfulness: {avg_faithfulness:.2f}")
    
    if avg_faithfulness < 0.8:
        print("❌ FALLO: La fidelidad del agente es baja. No desplegar a producción.")
    else:
        print("✅ ÉXITO: El agente cumple los estándares de calidad.")

if __name__ == "__main__":
    run_evaluation()
