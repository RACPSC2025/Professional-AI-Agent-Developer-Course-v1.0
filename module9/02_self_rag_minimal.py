"""
02_self_rag_minimal.py
======================
Implementación minimalista de la lógica Self-RAG (Self-Reflective RAG).
El agente genera una respuesta y luego se auto-critica para verificar
si alucinó o si la respuesta es relevante.

Métricas simuladas:
- IsRel (Relevance): ¿Es relevante el contexto?
- IsSup (Supported): ¿La respuesta está soportada por el contexto?

Requisitos:
pip install langchain langchain-openai
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 1. Componentes

def retrieve(query: str):
    """Simula un retriever"""
    # En prod, esto sería una búsqueda vectorial real
    knowledge_base = {
        "capital de francia": "La capital de Francia es París.",
        "capital de marte": "Marte no tiene capital conocida por humanos."
    }
    return knowledge_base.get(query.lower(), "No tengo información sobre eso.")

def generate(query: str, context: str):
    """Generador estándar"""
    prompt = ChatPromptTemplate.from_template(
        "Contexto: {context}\nPregunta: {query}\nRespuesta:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context})

def grade_relevance(query: str, context: str):
    """Crítico: Evalúa relevancia del contexto"""
    prompt = ChatPromptTemplate.from_template(
        "Pregunta: {query}\nContexto: {context}\n"
        "¿El contexto contiene información relevante para responder? Responde SOLO 'YES' o 'NO'."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context}).strip()

def grade_groundedness(answer: str, context: str):
    """Crítico: Evalúa si la respuesta está soportada (No alucinación)"""
    prompt = ChatPromptTemplate.from_template(
        "Contexto: {context}\nRespuesta: {answer}\n"
        "¿La respuesta está totalmente soportada por el contexto? Responde SOLO 'YES' o 'NO'."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"answer": answer, "context": context}).strip()

# 2. Pipeline Self-RAG

def self_rag_pipeline(query: str):
    print(f"🔍 Pregunta: {query}")
    
    # Paso 1: Retrieve
    context = retrieve(query)
    print(f"📄 Contexto recuperado: {context}")
    
    # Paso 2: Critique Retrieval (IsRel)
    relevance = grade_relevance(query, context)
    print(f"🤔 IsRel (Relevante): {relevance}")
    
    if relevance == "NO":
        return "❌ No pude encontrar información relevante para responder con seguridad."
    
    # Paso 3: Generate
    answer = generate(query, context)
    print(f"🤖 Respuesta generada: {answer}")
    
    # Paso 4: Critique Generation (IsSup)
    grounded = grade_groundedness(answer, context)
    print(f"🛡️ IsSup (Soportada): {grounded}")
    
    if grounded == "NO":
        # Aquí podríamos intentar regenerar o buscar de nuevo
        return "⚠️ La respuesta generada podría contener alucinaciones. Se ha bloqueado."
    
    return f"✅ Respuesta Final: {answer}"

def main():
    print("--- Test 1: Pregunta Conocida ---")
    print(self_rag_pipeline("Capital de Francia"))
    
    print("\n--- Test 2: Pregunta Desconocida (Simulando fallo de contexto) ---")
    # Simulamos que el retriever falló o trajo basura
    print(self_rag_pipeline("Capital de Atlantis"))

if __name__ == "__main__":
    main()
