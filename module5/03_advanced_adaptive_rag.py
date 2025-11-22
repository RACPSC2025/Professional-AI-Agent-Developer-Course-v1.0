"""
Módulo 5 - Ejemplo Avanzado: Adaptive Self-RAG
Framework: LangGraph
Caso de uso: Asistente de investigación científica con routing adaptativo

Self-RAG permite que el agente decida dinámicamente:
1. ¿Necesito recuperar información externa? (retrieval)
2. ¿Qué estrategia de búsqueda usar? (vectorial, keyword, híbrida)
3. ¿La respuesta generada es suficiente o necesito iterar?

Instalación:
pip install langgraph langchain langchain-openai langchain-community chromadb rank-bm25
"""

import os
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# Configuración
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
EMBEDDINGS = OpenAIEmbeddings()


class AdaptiveRAGState(TypedDict):
    """Estado del sistema Adaptive Self-RAG"""
    question: str
    query_type: Literal["factual", "conceptual", "analytical"]  # Tipo de consulta
    retrieval_strategy: Literal["vector", "keyword", "hybrid"]  # Estrategia elegida
    need_retrieval: bool  # ¿Necesita buscar información?
    documents: List[Document]
    generation: str
    confidence: float  # Confianza en la respuesta
    iteration_count: int


def create_research_corpus() -> tuple:
    """Crear corpus de investigación científica simulado"""
    papers = [
        "Attention Is All You Need (Vaswani et al., 2017): Introducimos el Transformer, "
        "una arquitectura de red neuronal basada completamente en mecanismos de atención, "
        "eliminando la recurrencia. Los resultados en traducción automática superan a RNNs.",
        
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018): "
        "BERT pre-entrena representaciones bidireccionales mediante masked language modeling. "
        "Logra state-of-the-art en 11 tareas de NLP.",
        
        "GPT-3: Language Models are Few-Shot Learners (Brown et al., 2020): "
        "Demostramos que modelos de lenguaje de 175B parámetros pueden realizar tareas "
        "mediante ejemplos en el prompt, sin fine-tuning.",
        
        "Chain-of-Thought Prompting (Wei et al., 2022): Generar razonamientos intermedios "
        "mejora significativamente el desempeño en tareas de razonamiento complejo.",
        
        "ReAct: Synergizing Reasoning and Acting (Yao et al., 2023): Combinar razonamiento "
        "en lenguaje natural con acciones permite a LLMs interactuar con entornos externos.",
        
        "Retrieval-Augmented Generation (Lewis et al., 2020): RAG combina recuperación de "
        "información con generación de texto, reduciendo alucinaciones y mejorando factualidad.",
        
        "LoRA: Low-Rank Adaptation (Hu et al., 2021): Fine-tuning eficiente mediante "
        "adaptación de bajo rango, reduciendo parámetros entrenables en 10,000x.",
        
        "Constitutional AI (Bai et al., 2022): Entrenar modelos para ser honestos, útiles "
        "e inofensivos mediante auto-crítica y refinamiento iterativo."
    ]
    
    # Preparar documentos
    documents = [Document(page_content=paper) for paper in papers]
    
    # Vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=EMBEDDINGS,
        collection_name="research_papers"
    )
    
    # BM25 retriever
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5
    
    return vectorstore, bm25_retriever, documents


def analyze_query_type(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Paso 1: Análisis del tipo de consulta"""
    print(f"\n🔍 Analizando tipo de consulta: '{state['question']}'")
    
    analysis_prompt = f"""Analiza la siguiente pregunta y clasifícala en una de estas categorías:

1. FACTUAL: Pregunta que requiere hechos específicos, números, nombres, fechas.
   Ejemplo: "¿Cuántos parámetros tiene GPT-3?"
   
2. CONCEPTUAL: Pregunta sobre conceptos, definiciones, explicaciones.
   Ejemplo: "¿Qué es el mecanismo de atención?"
   
3. ANALYTICAL: Pregunta que requiere análisis, comparación, razonamiento profundo.
   Ejemplo: "¿Por qué BERT supera a GPT en tareas de clasificación?"

Pregunta: {state['question']}

Responde SOLO con: factual, conceptual o analytical"""
    
    response = LLM.invoke(analysis_prompt)
    query_type = response.content.strip().lower()
    
    print(f"✅ Tipo identificado: {query_type.upper()}")
    
    return {**state, "query_type": query_type}


def decide_retrieval_need(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Paso 2: Decidir si necesita retrieval (Self-RAG)"""
    print("\n🤔 ¿Necesita recuperar información externa?")
    
    # El modelo decide si puede responder sin retrieval
    decision_prompt = f"""¿Necesitas buscar información en documentos externos para responder esta pregunta?

Pregunta: {state['question']}
Tipo: {state['query_type']}

Si la pregunta es de conocimiento general o puedes inferir la respuesta, di NO.
Si requiere datos específicos de papers, di SI.

Responde SOLO con: SI o NO"""
    
    response = LLM.invoke(decision_prompt)
    decision = response.content.strip().upper()
    
    need_retrieval = decision == "SI"
    print(f"✅ Decisión: {'RETRIEVAL NECESARIO' if need_retrieval else 'GENERAR DIRECTAMENTE'}")
    
    return {**state, "need_retrieval": need_retrieval}


def select_retrieval_strategy(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Paso 3: Seleccionar estrategia de retrieval adaptativa"""
    print(f"\n🎯 Seleccionando estrategia de retrieval para tipo: {state['query_type']}")
    
    # Routing basado en tipo de consulta
    strategy_map = {
        "factual": "keyword",      # Hechos específicos -> BM25 (keywords)
        "conceptual": "vector",    # Conceptos -> Embeddings semánticos
        "analytical": "hybrid"     # Análisis -> Híbrido (lo mejor de ambos)
    }
    
    strategy = strategy_map.get(state["query_type"], "hybrid")
    print(f"✅ Estrategia seleccionada: {strategy.upper()}")
    
    return {**state, "retrieval_strategy": strategy}


def retrieve_adaptive(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Paso 4: Recuperación adaptativa según estrategia"""
    print(f"\n📚 Ejecutando retrieval con estrategia: {state['retrieval_strategy']}")
    
    vectorstore, bm25_retriever, all_docs = create_research_corpus()
    question = state["question"]
    
    if state["retrieval_strategy"] == "vector":
        # Búsqueda vectorial (semántica)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        documents = retriever.get_relevant_documents(question)
        print("  🔹 Usando búsqueda VECTORIAL (semántica)")
        
    elif state["retrieval_strategy"] == "keyword":
        # Búsqueda BM25 (keywords)
        documents = bm25_retriever.get_relevant_documents(question)[:4]
        print("  🔹 Usando búsqueda KEYWORD (BM25)")
        
    else:  # hybrid
        # Ensemble: 70% vector + 30% keywords
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        documents = ensemble.get_relevant_documents(question)[:4]
        print("  🔹 Usando búsqueda HÍBRIDA (70% vector + 30% keyword)")
    
    print(f"✅ Recuperados {len(documents)} documentos")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc.page_content[:70]}...")
    
    return {**state, "documents": documents}


def generate_with_confidence(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Paso 5: Generar respuesta con auto-evaluación de confianza"""
    print("\n🤖 Generando respuesta...")
    
    # Contexto (si hay retrieval)
    context = ""
    if state["need_retrieval"] and state["documents"]:
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        context = f"\nContexto de papers:\n{context}\n"
    
    generation_prompt = f"""Eres un asistente de investigación científica.{context}
Pregunta: {state['question']}

Proporciona:
1. Una respuesta clara y precisa
2. Tu nivel de confianza (0-100%) en la respuesta

Formato:
RESPUESTA: [tu respuesta aquí]
CONFIANZA: [número del 0-100]"""
    
    response = LLM.invoke(generation_prompt)
    content = response.content
    
    # Parsear respuesta y confianza
    lines = content.split("\n")
    answer_lines = []
    confidence = 50.0  # default
    
    for line in lines:
        if line.startswith("CONFIANZA:"):
            try:
                confidence = float(line.split(":")[1].strip().replace("%", ""))
            except:
                pass
        elif line.startswith("RESPUESTA:"):
            answer_lines.append(line.replace("RESPUESTA:", "").strip())
        elif answer_lines:  # Continuar capturando la respuesta
            answer_lines.append(line)
    
    answer = "\n".join(answer_lines).strip()
    
    print(f"\n✅ Respuesta generada (Confianza: {confidence}%)")
    
    return {**state, "generation": answer, "confidence": confidence}


def should_retrieve(state: AdaptiveRAGState) -> str:
    """Decisión: ¿Necesitamos retrieval?"""
    return "retrieve" if state["need_retrieval"] else "generate"


def should_iterate(state: AdaptiveRAGState) -> str:
    """Decisión: ¿Necesitamos iterar? (si confianza es baja)"""
    # Si la confianza es muy baja, podríamos iterar (simplificado aquí)
    if state.get("confidence", 100) < 40 and state["iteration_count"] == 0:
        return "iterate"
    return "end"


def create_adaptive_rag_graph():
    """Construir grafo de Self-RAG Adaptativo"""
    workflow = StateGraph(AdaptiveRAGState)
    
    # Nodos
    workflow.add_node("analyze", analyze_query_type)
    workflow.add_node("decide", decide_retrieval_need)
    workflow.add_node("strategy", select_retrieval_strategy)
    workflow.add_node("retrieve", retrieve_adaptive)
    workflow.add_node("generate", generate_with_confidence)
    
    # Flujo
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "decide")
    
    # Decisión: ¿retrieval o generar directamente?
    workflow.add_conditional_edges(
        "decide",
        should_retrieve,
        {
            "retrieve": "strategy",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("strategy", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def main():
    """Demostración de Adaptive Self-RAG"""
    print("=" * 80)
    print("Asistente de Investigación Científica - Adaptive Self-RAG")
    print("=" * 80)
    
    app = create_adaptive_rag_graph()
    
    # Casos de prueba que demuestran diferentes estrategias
    test_questions = [
        "¿Cuántos parámetros tiene GPT-3?",  # Factual -> keyword search
        "¿Qué es Chain-of-Thought?",  # Conceptual -> vector search
        "¿Por qué RAG reduce las alucinaciones comparado con LLMs puros?",  # Analytical -> hybrid
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"PREGUNTA {i}: {question}")
        print("=" * 80)
        
        initial_state = AdaptiveRAGState(
            question=question,
            query_type="",
            retrieval_strategy="",
            need_retrieval=False,
            documents=[],
            generation="",
            confidence=0.0,
            iteration_count=0
        )
        
        result = app.invoke(initial_state)
        
        print(f"\n📊 RESULTADOS:")
        print(f"  - Tipo de consulta: {result['query_type'].upper()}")
        print(f"  - Retrieval usado: {'SÍ' if result['need_retrieval'] else 'NO'}")
        if result['need_retrieval']:
            print(f"  - Estrategia: {result['retrieval_strategy'].upper()}")
        print(f"  - Confianza: {result['confidence']}%")
        print(f"\n📝 RESPUESTA:")
        print(f"{result['generation']}")
        print("=" * 80)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada en .env")
    
    main()
