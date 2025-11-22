"""
Módulo 5 - Ejemplo Intermedio: Corrective RAG (CRAG)
Framework: LangGraph
Caso de uso: Sistema de soporte técnico que auto-corrige búsquedas irrelevantes

Corrective RAG evalúa la calidad de los documentos recuperados y, si no son relevantes,
busca en fuentes externas (web search) o reformula la consulta.

Instalación:
pip install langgraph langchain langchain-openai langchain-community chromadb tavily-python
"""

import os
from typing import TypedDict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# Configuración
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
EMBEDDINGS = OpenAIEmbeddings()


class CRAGState(TypedDict):
    """Estado del grafo de Corrective RAG"""
    question: str
    documents: List[Document]
    generation: str
    relevance_score: str  # "relevant" o "irrelevant"
    web_search_needed: bool


def create_knowledge_base() -> Chroma:
    """Crear base de conocimientos de soporte técnico"""
    docs = [
        "Para resetear tu password, ve a Configuración > Seguridad > Cambiar Password. "
        "Necesitarás tu email de verificación.",
        
        "El error 'Connection Timeout' generalmente indica problemas de red. "
        "Verifica tu firewall y asegúrate de que el puerto 443 esté abierto.",
        
        "Para exportar tus datos, usa el botón 'Exportar' en el panel principal. "
        "Soportamos formatos CSV, JSON y Excel.",
        
        "Si la aplicación se cierra inesperadamente, revisa los logs en C:\\AppLogs. "
        "Busca mensajes con nivel ERROR o FATAL.",
        
        "Para actualizar a la versión premium, ve a Cuenta > Suscripción. "
        "Aceptamos tarjetas de crédito y PayPal."
    ]
    
    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = [Document(page_content=doc) for doc in docs]
    splits = text_splitter.split_documents(documents)
    
    # Crear vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=EMBEDDINGS,
        collection_name="tech_support"
    )
    return vectorstore


def retrieve_documents(state: CRAGState) -> CRAGState:
    """Paso 1: Recuperar documentos del vectorstore"""
    print(f"\n📚 Recuperando documentos para: '{state['question']}'")
    
    vectorstore = create_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    documents = retriever.get_relevant_documents(state["question"])
    
    print(f"✅ Recuperados {len(documents)} documentos")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc.page_content[:80]}...")
    
    return {**state, "documents": documents}


def grade_documents(state: CRAGState) -> CRAGState:
    """Paso 2: Evaluar relevancia de los documentos (LLM-as-a-Judge)"""
    print("\n⚖️ Evaluando relevancia de documentos...")
    
    # Concatenar contenido de documentos
    docs_content = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    # Prompt para evaluación
    grading_prompt = f"""Eres un evaluador de relevancia. Analiza si los siguientes documentos 
son relevantes para responder la pregunta del usuario.

Pregunta: {state['question']}

Documentos:
{docs_content}

¿Los documentos contienen información relevante para responder la pregunta?
Responde SOLO con 'relevant' o 'irrelevant'."""
    
    response = LLM.invoke(grading_prompt)
    relevance = response.content.strip().lower()
    
    if relevance == "relevant":
        print("✅ Documentos RELEVANTES - Proceder a generar respuesta")
        return {**state, "relevance_score": "relevant", "web_search_needed": False}
    else:
        print("❌ Documentos IRRELEVANTES - Se requiere búsqueda web")
        return {**state, "relevance_score": "irrelevant", "web_search_needed": True}


def web_search(state: CRAGState) -> CRAGState:
    """Paso 3a: Búsqueda web si los documentos no son relevantes"""
    print("\n🌐 Realizando búsqueda web complementaria...")
    
    # Tavily para búsqueda web
    web_search_tool = TavilySearchResults(max_results=3)
    search_results = web_search_tool.invoke({"query": state["question"]})
    
    # Convertir resultados a Documents
    web_docs = [
        Document(
            page_content=result.get("content", ""),
            metadata={"source": result.get("url", "unknown")}
        )
        for result in search_results
    ]
    
    print(f"✅ Encontrados {len(web_docs)} resultados web")
    
    # Combinar con documentos originales
    all_documents = state["documents"] + web_docs
    
    return {**state, "documents": all_documents, "relevance_score": "relevant"}


def generate_answer(state: CRAGState) -> CRAGState:
    """Paso 3b/4: Generar respuesta usando documentos relevantes"""
    print("\n🤖 Generando respuesta final...")
    
    # Context from documents
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    generation_prompt = f"""Eres un asistente de soporte técnico. Responde la pregunta del usuario 
basándote ÚNICAMENTE en el contexto proporcionado.

Contexto:
{context}

Pregunta: {state['question']}

Respuesta (si no hay información suficiente, dilo claramente):"""
    
    response = LLM.invoke(generation_prompt)
    answer = response.content.strip()
    
    print(f"\n✅ Respuesta generada:\n{answer}")
    
    return {**state, "generation": answer}


def should_web_search(state: CRAGState) -> str:
    """Decisión: ¿Necesitamos búsqueda web?"""
    if state.get("web_search_needed", False):
        return "web_search"
    else:
        return "generate"


# Construcción del grafo de Corrective RAG
def create_crag_graph():
    """Construir el grafo de flujo de Corrective RAG"""
    workflow = StateGraph(CRAGState)
    
    # Nodos
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate_answer)
    
    # Flujo
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    
    # Decisión condicional: ¿web search o generar directamente?
    workflow.add_conditional_edges(
        "grade",
        should_web_search,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def main():
    """Función principal para demostrar Corrective RAG"""
    print("=" * 70)
    print("Sistema de Soporte Técnico con Corrective RAG")
    print("=" * 70)
    
    # Compilar grafo
    app = create_crag_graph()
    
    # Test cases
    test_questions = [
        "¿Cómo reseteo mi password?",  # Debe ser respondida con docs locales
        "¿Cuáles son las últimas noticias de OpenAI?",  # Requiere web search
        "¿Cómo exporto mis datos?"  # Debe ser respondida con docs locales
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"PREGUNTA {i}: {question}")
        print("=" * 70)
        
        # Ejecutar grafo
        initial_state = CRAGState(
            question=question,
            documents=[],
            generation="",
            relevance_score="",
            web_search_needed=False
        )
        
        result = app.invoke(initial_state)
        
        print(f"\n📝 RESPUESTA FINAL:")
        print(f"{result['generation']}")
        print(f"\n🔍 Fuentes usadas: {'Internal KB + Web' if result['web_search_needed'] else 'Internal KB only'}")
        print("=" * 70)


if __name__ == "__main__":
    # Verificar variables de entorno
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("❌ TAVILY_API_KEY no configurada")
    
    main()
