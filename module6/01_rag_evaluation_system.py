"""
🟢 NIVEL BÁSICO: SISTEMA DE EVALUACIÓN RAG CON RAGAS
----------------------------------------------------
Este ejemplo demuestra cómo evaluar la calidad de un sistema RAG usando métricas profesionales.
Caso de Uso: Validación automatizada de un chatbot de soporte técnico.

Conceptos Clave:
- Ragas: Framework de evaluación RAG (Faithfulness, Answer Relevancy, Context Recall).
- LLM-as-a-Judge: Usar GPT-4 para evaluar respuestas.
- Test datasets con ground truth.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import json

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. PREPARAR SISTEMA RAG (MOCK) ---
# Simulamos un vectorstore simple para el ejemplo
sample_kb = """
MANUAL DE TROUBLESHOOTING - IMPRESORA XP-500

PROBLEMA: La impresora no enciende
SOLUCIÓN: 
1. Verificar que el cable de alimentación esté conectado.
2. Revisar que el interruptor trasero esté en posición ON.
3. Probar con otro enchufe.

PROBLEMA: Atascos de papel frecuentes
SOLUCIÓN:
1. Revisar que el papel sea del tamaño correcto (A4 o Letter).
2. No sobrecargar la bandeja (máximo 100 hojas).
3. Limpiar los rodillos con un paño húmedo.

PROBLEMA: Impresión con rayas o manchas
SOLUCIÓN:
1. Ejecutar ciclo de limpieza de cabezales (Menú > Mantenimiento > Limpiar).
2. Verificar niveles de tinta (mínimo 30%).
3. Reemplazar cartuchos si tienen más de 6 meses.
"""

# Crear vectorstore simple
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.create_documents([sample_kb])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings, collection_name="printer_kb")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# RAG Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Eres un asistente de soporte técnico para impresoras.
Usa SOLO el contexto proporcionado para responder.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:
""")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- 2. DATASET DE EVALUACIÓN ---
# En producción, esto vendría de tickets reales resueltos
eval_dataset = [
    {
        "question": "Mi impresora no enciende, ¿qué hago?",
        "ground_truth": "Verificar cable de alimentación, revisar interruptor trasero en ON, probar otro enchufe.",
        "context_needed": ["cable de alimentación", "interruptor"]
    },
    {
        "question": "¿Cómo soluciono los atascos de papel?",
        "ground_truth": "Usar papel correcto (A4/Letter), no sobrecargar bandeja (máximo 100 hojas), limpiar rodillos.",
        "context_needed": ["papel", "bandeja", "rodillos"]
    },
    {
        "question": "La impresión sale con rayas, ¿qué hago?",
        "ground_truth": "Ejecutar limpieza de cabezales, verificar niveles de tinta (mínimo 30%), reemplazar cartuchos viejos.",
        "context_needed": ["limpieza", "cabezales", "tinta"]
    }
]

# --- 3. EVALUACIÓN MANUAL (SIN RAGAS) ---
# Implementamos métricas básicas manualmente para entender los conceptos

def evaluate_faithfulness(answer: str, contexts: List[str], llm) -> float:
    """
    Evalúa si la respuesta se basa en el contexto (no alucina).
    1.0 = Perfectamente fiel, 0.0 = Completamente alucinada.
    """
    eval_prompt = f"""
Evalúa si la RESPUESTA se basa únicamente en el CONTEXTO proporcionado.
No debe contener información que no esté en el contexto.

CONTEXTO:
{' '.join(contexts)}

RESPUESTA:
{answer}

Responde SOLO con un número del 0.0 al 1.0:
- 1.0: La respuesta se basa completamente en el contexto
- 0.5: La respuesta es parcialmente correcta pero añade información
- 0.0: La respuesta alucina información no presente en el contexto

PUNTUACIÓN:
    """
    
    score_text = llm.invoke(eval_prompt).content.strip()
    try:
        return float(score_text)
    except:
        return 0.5  # Default si falla el parsing

def evaluate_relevancy(answer: str, question: str, llm) -> float:
    """
    Evalúa si la respuesta es relevante para la pregunta.
    """
    eval_prompt = f"""
Evalúa qué tan relevante es la RESPUESTA para la PREGUNTA.

PREGUNTA: {question}

RESPUESTA: {answer}

Responde SOLO con un número del 0.0 al 1.0:
- 1.0: Respuesta perfectamente relevante y útil
- 0.5: Respuesta parcialmente relevante
- 0.0: Respuesta irrelevante

PUNTUACIÓN:
    """
    
    score_text = llm.invoke(eval_prompt).content.strip()
    try:
        return float(score_text)
    except:
        return 0.5

def evaluate_context_recall(contexts: List[str], ground_truth: str, llm) -> float:
    """
    Evalúa si el contexto recuperado contiene la información necesaria.
    """
    eval_prompt = f"""
Evalúa si el CONTEXTO contiene la información necesaria para responder correctamente.

CONTEXTO RECUPERADO:
{' '.join(contexts)}

RESPUESTA CORRECTA ESPERADA:
{ground_truth}

Responde SOLO con un número del 0.0 al 1.0:
- 1.0: El contexto contiene toda la información necesaria
- 0.5: El contexto contiene información parcial
- 0.0: El contexto no contiene la información necesaria

PUNTUACIÓN:
    """
    
    score_text = llm.invoke(eval_prompt).content.strip()
    try:
        return float(score_text)
    except:
        return 0.5

# --- 4. EJECUCIÓN DE EVALUACIÓN ---
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Modelo más potente como juez

print("="*70)
print("  📊 SISTEMA DE EVALUACIÓN RAG - RAGAS METRICS")
print("="*70)
print("\n🔍 Evaluando sistema RAG con dataset de prueba...\n")

results = []

for i, item in enumerate(eval_dataset):
    print(f"--- TEST CASE {i+1}/{len(eval_dataset)} ---")
    print(f"Pregunta: {item['question']}")
    
    # Generar respuesta
    retrieved_docs = retriever.get_relevant_documents(item['question'])
    contexts = [doc.page_content for doc in retrieved_docs]
    answer = rag_chain.invoke(item['question'])
    
    print(f"Respuesta generada: {answer[:100]}...")
    
    # Evaluar métricas
    faithfulness = evaluate_faithfulness(answer, contexts, judge_llm)
    relevancy = evaluate_relevancy(answer, item['question'], judge_llm)
    context_recall = evaluate_context_recall(contexts, item['ground_truth'], judge_llm)
    
    print(f"\n📈 MÉTRICAS:")
    print(f"   Faithfulness (Fidelidad): {faithfulness:.2f}")
    print(f"   Answer Relevancy (Relevancia): {relevancy:.2f}")
    print(f"   Context Recall (Recuperación): {context_recall:.2f}")
    print()
    
    results.append({
        "question": item['question'],
        "answer": answer,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "context_recall": context_recall
    })

# --- 5. RESUMEN FINAL ---
avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
avg_relevancy = sum(r['relevancy'] for r in results) / len(results)
avg_context_recall = sum(r['context_recall'] for r in results) / len(results)

print("="*70)
print("  📊 RESUMEN DE EVALUACIÓN")
print("="*70)
print(f"Casos evaluados: {len(results)}")
print(f"\nPROMEDIOS:")
print(f"  Faithfulness: {avg_faithfulness:.2f} (Target: >0.85)")
print(f"  Relevancy: {avg_relevancy:.2f} (Target: >0.80)")
print(f"  Context Recall: {avg_context_recall:.2f} (Target: >0.90)")

# Recomendaciones
print(f"\n💡 RECOMENDACIONES:")
if avg_faithfulness < 0.85:
    print("  ⚠️ BAJA FIDELIDAD: El modelo tiende a alucinar. Ajustar prompt para que sea más estricto.")
if avg_relevancy < 0.80:
    print("  ⚠️ BAJA RELEVANCIA: Las respuestas no son útiles. Revisar calidad del contexto.")
if avg_context_recall < 0.90:
    print("  ⚠️ BAJA RECUPERACIÓN: El retriever no encuentra documentos relevantes. Mejorar chunking o embeddings.")

if avg_faithfulness >= 0.85 and avg_relevancy >= 0.80 and avg_context_recall >= 0.90:
    print("  ✅ SISTEMA RAG EN BUENAS CONDICIONES para producción.")

# Guardar resultados
with open("rag_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n💾 Resultados guardados en: rag_evaluation_results.json")
