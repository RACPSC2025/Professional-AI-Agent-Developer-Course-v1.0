# Módulo 5: RAG Avanzado (The Memory)

## 🎯 Objetivos del Módulo
El RAG básico (dividir texto -> vectorizar -> buscar) ya no es suficiente para producción. En este módulo, implementaremos técnicas avanzadas de **Retrieval-Augmented Generation** para manejar documentos complejos, tablas y relaciones semánticas profundas.

## 📚 Conceptos Clave (The RAG Stack)

### 1. Ingesta y Chunking Inteligente
-   **Semantic Chunking:** Dividir por significado, no por caracteres.
-   **Propositional Chunking:** Convertir oraciones complejas en proposiciones atómicas.
-   **Document Parsing:** Extraer tablas, imágenes y metadata de PDFs complejos.

### 2. Recuperación (Retrieval)
-   **Hybrid Search:** Combinar búsqueda vectorial (semántica) con BM25 (palabras clave exactas).
-   **Reranking:** Usar un modelo Cross-Encoder (ej. Cohere) para reordenar los resultados y mejorar la precisión.
-   **Multi-Index:** Diferentes índices para diferentes tipos de contenido (código, docs, tablas).

### 3. Transformación de Consultas
-   **Query Rewriting:** Reformular la pregunta del usuario para mejorar resultados.
-   **Multi-Query:** Generar múltiples variantes de la consulta y fusionar resultados.
-   **HyDE (Hypothetical Document Embeddings):** Generar un documento hipotético y buscar por él.

### 4. Corrective RAG (CRAG)
-   Evaluar la relevancia de los documentos recuperados.
-   Si no son relevantes, buscar en fuentes externas (web search).
-   Auto-corrección del proceso de retrieval.

### 5. Self-RAG y Adaptive RAG
-   **Self-RAG:** El modelo decide cuándo necesita información externa.
-   **Adaptive RAG:** Routing inteligente entre diferentes estrategias según el tipo de consulta.

## 💻 Snippet de Código: Hybrid Search con Reranking

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Retriever vectorial (semántico)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Retriever BM25 (keyword-based)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Ensemble (híbrido)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 50% cada uno
)

# Recuperar documentos
docs = ensemble_retriever.get_relevant_documents(query)

# Reranking con modelo cross-encoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank(model="rerank-english-v2.0", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# Resultados finales (top 3 más relevantes)
final_docs = compression_retriever.get_relevant_documents(query)
```

## 🛠️ Proyectos Prácticos

### 🟢 Nivel Básico: Hybrid Search RAG
**Archivo:** `01_hybrid_search_rag.py`
-   **Concepto:** Combinar búsqueda semántica y keyword-based.
-   **Framework:** LangChain
-   **Caso de uso:** Sistema de FAQ empresarial con documentación técnica.

### 🟡 Nivel Intermedio: Corrective RAG
**Archivo:** `02_intermediate_corrective_rag.py`
-   **Concepto:** Auto-corrección con evaluación de relevancia.
-   **Framework:** LangGraph
-   **Caso de uso:** Soporte técnico que verifica calidad de respuestas.

### 🔴 Nivel Avanzado: Adaptive Self-RAG
**Archivo:** `03_advanced_adaptive_rag.py`
-   **Concepto:** Routing adaptativo con múltiples estrategias.
-   **Framework:** LangGraph
-   **Caso de uso:** Asistente de investigación científica con fuentes especializadas.

## 🎓 Mejores Prácticas

1. **Siempre usar Hybrid Search** en producción (vectorial + BM25).
2. **Reranking es crucial:** Mejora precision hasta 30%.
3. **Medir métricas:** Context Precision, Context Recall, Answer Relevancy.
4. **Chunking inteligente:** El tamaño importa (experimenta entre 200-800 tokens).
5. **Metadata matters:** Añade fuente, fecha, categoría a cada chunk.
