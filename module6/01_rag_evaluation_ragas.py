"""
01_rag_evaluation_ragas.py
==========================
Este script demuestra cómo implementar un pipeline de evaluación profesional para sistemas RAG
utilizando el framework Ragas (Retrieval Augmented Generation Assessment).

Conceptos clave:
1. Generación de datos sintéticos (Testset Generation)
2. Métricas Ragas: Faithfulness, Answer Relevancy, Context Precision, Context Recall
3. Evaluación con "LLM-as-a-Judge"

Requisitos:
pip install ragas langchain langchain-openai datasets
"""

import os
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Configuración de API Keys (asegúrate de tenerlas en tu .env)
# os.environ["OPENAI_API_KEY"] = "sk-..."

def create_synthetic_testset(documents):
    """
    Genera un dataset de prueba sintético a partir de documentos.
    Ragas usa el LLM para crear preguntas y respuestas ground-truth basadas en tus docs.
    """
    print("🤖 Generando dataset sintético (esto puede tardar)...")
    
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()
    
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )
    
    # Generar 5 preguntas de prueba con diferentes tipos de complejidad
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=5,
        distributions={
            simple: 0.5,
            reasoning: 0.25,
            multi_context: 0.25
        }
    )
    
    return testset.to_pandas()

def run_evaluation(dataset_dict):
    """
    Ejecuta la evaluación Ragas sobre un dataset.
    
    El dataset debe tener las columnas:
    - question: La pregunta del usuario
    - answer: La respuesta generada por tu RAG
    - contexts: Los chunks recuperados por tu retriever
    - ground_truth: La respuesta correcta esperada
    """
    print("📊 Ejecutando evaluación Ragas...")
    
    # Convertir a formato HuggingFace Dataset
    hf_dataset = Dataset.from_dict(dataset_dict)
    
    # Definir métricas a evaluar
    metrics = [
        faithfulness,      # ¿La respuesta se basa en el contexto? (Evita alucinaciones)
        answer_relevancy,  # ¿La respuesta contesta a la pregunta?
        context_precision, # ¿Los chunks relevantes están al principio?
        context_recall,    # ¿Se recuperó toda la info necesaria?
    ]
    
    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
    )
    
    return results

def main():
    # 1. Cargar documentos de ejemplo (usamos un texto dummy para la demo)
    print("📂 Cargando documentos...")
    text = """
    La arquitectura Transformer fue introducida por Google en 2017 en el paper "Attention Is All You Need".
    A diferencia de las RNNs, los Transformers procesan toda la secuencia en paralelo gracias al mecanismo de Self-Attention.
    BERT (Bidirectional Encoder Representations from Transformers) es un modelo encoder-only lanzado en 2018.
    GPT (Generative Pre-trained Transformer) es un modelo decoder-only lanzado por OpenAI.
    Ragas es un framework para evaluar pipelines RAG usando métricas como Faithfulness y Context Recall.
    """
    
    # Simular carga de documentos
    from langchain.docstore.document import Document
    documents = [Document(page_content=text, metadata={"source": "ai_history.txt"})]
    
    # 2. (Opcional) Generar testset sintético
    # En un caso real, haríamos esto. Para la demo, usaremos datos hardcodeados para rapidez.
    # test_df = create_synthetic_testset(documents)
    
    # 3. Datos de prueba simulados (Simulando una ejecución de RAG)
    eval_data = {
        'question': [
            '¿Cuándo se introdujo Transformer?',
            '¿Qué diferencia a BERT de GPT?',
            '¿Qué es Ragas?'
        ],
        'answer': [
            'Google introdujo Transformer en 2017.', # Correcta y fiel
            'BERT es encoder-only y GPT es decoder-only.', # Correcta y fiel
            'Ragas es una comida tradicional de la India.' # ALUCINACIÓN (Falso)
        ],
        'contexts': [
            ['La arquitectura Transformer fue introducida por Google en 2017...'],
            ['BERT es un modelo encoder-only... GPT es un modelo decoder-only...'],
            ['Ragas es un framework para evaluar pipelines RAG...'] # El contexto dice la verdad
        ],
        'ground_truth': [
            'En 2017 por Google.',
            'BERT usa arquitectura encoder-only mientras que GPT usa decoder-only.',
            'Un framework para evaluar sistemas RAG.'
        ]
    }
    
    # 4. Ejecutar evaluación
    results = run_evaluation(eval_data)
    
    # 5. Mostrar resultados
    print("\n📈 Resultados de Evaluación:")
    print(results)
    
    df_results = results.to_pandas()
    print("\n📋 Detalle por ejemplo:")
    print(df_results[['question', 'faithfulness', 'answer_relevancy']].to_markdown())
    
    # Análisis
    print("\n🔍 Análisis:")
    print("Nota como la 3ra pregunta debería tener baja 'faithfulness' porque la respuesta (comida)")
    print("no coincide con el contexto (framework), aunque gramaticalmente sea correcta.")

if __name__ == "__main__":
    main()
