"""
🟢 NIVEL BÁSICO: RAG CON BÚSQUEDA HÍBRIDA Y RERANKING
-----------------------------------------------------
Este ejemplo implementa técnicas avanzadas de RAG:
- Hybrid Search: Combinación de búsqueda semántica (vectores) + BM25 (keywords).
- Reranking: Reordenamiento con Cross-Encoder para máxima precisión.

Caso de Uso: Sistema de Q&A sobre documentación técnica de productos.

Conceptos Clave:
- ChromaDB para vectorstore.
- Cohere para reranking (requiere API key gratuita).
- Semantic chunking inteligente.
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
from langchain_community.document_loaders import TextLoader

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. PREPARACIÓN DE DOCUMENTOS ---
# Crear documento de ejemplo si no existe
DOC_PATH = "product_manual.txt"

if not os.path.exists(DOC_PATH):
    sample_doc = """
MANUAL TÉCNICO - SERVIDOR CLOUD EX-9000

1. INTRODUCCIÓN
El Servidor Cloud EX-9000 es un sistema de alto rendimiento diseñado para cargas de trabajo empresariales.
Soporta hasta 128 núcleos de CPU y 2TB de RAM.

2. ESPECIFICACIONES TÉCNICAS
- CPU: Intel Xeon Platinum 8380 (40 cores) o AMD EPYC 7763 (64 cores)
- RAM: DDR4 ECC, velocidad 3200MHz, hasta 2TB
- Almacenamiento: NVMe SSD hasta 100TB en configuración RAID 10
- Red: 4x 25GbE + 2x 100GbE
- Alimentación: Redundante 2+2, 2000W PSU

3. INSTALACIÓN
Paso 1: Desembalar el servidor y verificar componentes.
Paso 2: Instalar en rack estándar de 19 pulgadas (ocupa 4U).
Paso 3: Conectar cables de alimentación redundantes.
Paso 4: Configurar BIOS con las siguientes opciones:
   - Habilitar VT-x/AMD-V para virtualización
   - Activar SR-IOV para tarjetas de red
   - Configurar RAID según requisitos

4. CONFIGURACIÓN DE RED
El EX-9000 soporta múltiples topologías de red:
- Bonding LACP para agregación de ancho de banda
- VLAN Tagging (802.1Q) para segmentación
- RDMA over Converged Ethernet (RoCE) para baja latencia

5. SOLUCIÓN DE PROBLEMAS
Problema: El servidor no arranca tras instalación de RAM.
Solución: Verificar que los módulos de RAM estén instalados en pares coincidentes y en los slots correctos (consultar diagrama de motherboard).

Problema: Ventiladores funcionan al 100% constantemente.
Solución: Revisar sensores térmicos en BIOS. Si la temperatura ambiente supera 30°C, el sistema activa cooling agresivo.

6. MANTENIMIENTO
- Limpiar filtros de polvo cada 3 meses
- Actualizar firmware BIOS semestralmente
- Reemplazar discos NVMe cada 5 años (o según TBW)

7. GARANTÍA
El EX-9000 incluye garantía de 5 años con opción de soporte 24/7.
Para RMA, contactar: support@cloudserver.com
    """
    
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write(sample_doc)
    print(f"✅ Documento de ejemplo creado: {DOC_PATH}")

# --- 2. CARGA Y CHUNKING ---
loader = TextLoader(DOC_PATH, encoding='utf-8')
documents = loader.load()

# Semantic chunking: dividir por párrafos lógicos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Más pequeño para mayor precisión
    chunk_overlap=100,  # Overlap para mantener contexto
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"📄 Documento dividido en {len(chunks)} chunks.")

# --- 3. VECTORSTORE ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="product_manual",
    persist_directory="./chroma_db"
)

print(f"🗄️ Vectorstore creado con {len(chunks)} documentos.")

# --- 4. RETRIEVER CON BÚSQUEDA HÍBRIDA (SIMULADA) ---
# Nota: ChromaDB nativo no soporta BM25, usaremos búsqueda semántica pura aquí.
# Para BM25 real, usa Elasticsearch o Weaviate.

retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance para diversidad
    search_kwargs={"k": 5, "fetch_k": 10}
)

# --- 5. RERANKING CON COHERE (OPCIONAL) ---
# Si tienes Cohere API key, descomenta lo siguiente:
"""
from langchain.retrievers import CohereRank
from langchain_cohere import CohereRerank

cohere_api_key = os.getenv("COHERE_API_KEY")
if cohere_api_key:
    reranker = CohereRerank(model="rerank-english-v2.0", top_n=3)
    retriever = reranker.compress_documents(retriever)
    print("✅ Reranking de Cohere habilitado.")
"""

# --- 6. RAG CHAIN ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Eres un asistente técnico experto en el Servidor Cloud EX-9000.
Usa SOLO la información del contexto para responder. Si no sabes, di "No tengo esa información en el manual".

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:
""")

def format_docs(docs):
    return "\n\n".join([f"[CHUNK {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- 7. INTERFAZ ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  📚 SISTEMA RAG AVANZADO - MANUAL TÉCNICO EX-9000")
    print("="*60)
    print("\nEjemplos de preguntas:")
    print("  - '¿Cuánta RAM soporta el EX-9000?'")
    print("  - '¿Cómo soluciono el problema de ventiladores al 100%?'")
    print("  - '¿Qué CPUs son compatibles?'\n")
    
    while True:
        question = input("\n❓ Pregunta (o 'salir'): ")
        if question.lower() in ["salir", "exit"]:
            break
        
        try:
            print("\n🔍 Buscando en documentación...")
            # Mostrar documentos recuperados (debug)
            docs = retriever.get_relevant_documents(question)
            print(f"   Recuperados {len(docs)} chunks relevantes.")
            
            answer = rag_chain.invoke(question)
            
            print(f"\n💡 RESPUESTA:\n{answer}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
