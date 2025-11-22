# Troubleshooting Guide - AI Agent Development

## 🎯 Top 20 Problemas Comunes y Soluciones

Esta guía cubre los problemas más frecuentes al desarrollar AI Agents y sus soluciones.

---

## 1. Instalación y Setup

### ❌ Problema: "Module not found" después de pip install

**Síntomas:**
```
ImportError: No module named 'langchain'
```

**Causas Comunes:**
- Virtual environment no activado
- Instalado en Python incorrecto (sistema vs venv)
- Path issues

**Solución:**
```bash
# Verificar qué Python estás usando
which python  # macOS/Linux
where python  # Windows

# Activar venv correctamente
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Reinstalar en el venv correcto
pip install langchain langchain-openai

# Verificar
python -c "import langchain; print(langchain.__version__)"
```

---

### ❌ Problema: Conflictos de versiones

**Síntomas:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solución:**
```bash
# Crear venv limpio
python -m venv venv_fresh
source venv_fresh/bin/activate

# Instalar requirements.txt específicas
pip install -r requirements.txt

# Si sigue fallando, instalar una por una
pip install langchain==0.1.0
pip install langchain-openai==0.0.5
```

---

## 2. API y Autenticación

### ❌ Problema: "AuthenticationError: Incorrect API key"

**Síntomas:**
```
openai.error.AuthenticationError: Incorrect API key provided
```

**Causas:**
- API key incorrecta o expirada
- Variable de entorno no cargada
- Typo en el nombre de la variable

**Solución:**
```python
# Verificar que .env existe y tiene la key
# .env
OPENAI_API_KEY=sk-proj-abc123...

# Cargar correctamente
from dotenv import load_dotenv
import os

load_dotenv()  # ⚠️ Debe ser ANTES de importar OpenAI

api_key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded: {api_key[:20]}...")  # Debug

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
```

---

### ❌ Problema: Rate Limiting / "Rate limit exceeded"

**Síntomas:**
```
openai.error.RateLimitError: Rate limit reached
```

**Soluciones:**

**Opción 1: Exponential Backoff**
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_llm(prompt):
    return llm.invoke(prompt)
```

**Opción 2: Rate Limiter**
```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=50, period=60)  # 50 calls per minute
def call_llm(prompt):
    return llm.invoke(prompt)
```

**Opción 3: Batch Processing**
```python
# Procesar en batches pequeños
def process_batch(items, batch_size=10):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        results = llm.batch(batch)
        time.sleep(1)  # Cooldown
        yield results
```

---

## 3. Performance

### ❌ Problema: Respuestas muy lentas (>10 segundos)

**Causas:**
- Modelo demasiado grande
- Context muy largo
- No streaming
- Llamadas síncronas

**Soluciones:**

**1. Model Downgrade**
```python
# De esto
llm = ChatOpenAI(model="gpt-4")  # Slow

# A esto
llm = ChatOpenAI(model="gpt-4o-mini")  # 10x faster
```

**2. Streaming**
```python
# Mejorar perceived latency
for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)
```

**3. Reduce Context**
```python
# Limitar tokens
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=2000)  # No todo el doc
relevant_chunks = splitter.split_text(long_document)[:3]
```

**4. Parallel Calls**
```python
import asyncio

async def process_parallel(items):
    tasks = [llm.ainvoke(item) for item in items]
    return await asyncio.gather(*tasks)

# 5 items en 2 segundos en vez de 10
```

---

### ❌ Problema: Memory Leaks / Out of Memory

**Síntomas:**
```
MemoryError: Unable to allocate array
```

**Causas:**
- Vectorstore en memoria muy grande
- No limpiar embeddings cache
- Loops sin límite

**Soluciones:**
```python
# 1. Usar disk-based vectorstore
vectorstore = Chroma(
    persist_directory="./chroma_db",  # No en RAM
    embedding_function=embeddings
)

# 2. Limpiar cache periódicamente
import gc

def process_large_dataset(items):
    for i, item in enumerate(items):
        process(item)
        
        if i % 100 == 0:
            gc.collect()  # Force garbage collection

# 3. Usar generators
def load_documents():
    for file in files:
        yield load_file(file)  # No cargar todo en RAM
```

---

## 4. Calidad de Respuestas

### ❌ Problema: Respuestas genéricas o irrelevantes

**Causa:** Prompt poco específico

**Solución:**
```python
# ❌ Malo
prompt = "Analiza esto"

# ✅ Bueno
prompt = """Eres un analista financiero senior.

Analiza el siguiente estado financiero Q3 2024:
{financial_data}

Proporciona:
1. Resumen ejecutivo (2-3 líneas)
2. Métricas clave (deuda/equity, ROE, margen operativo)
3. Red flags si los hay
4. Recomendación: comprar/vender/mantener

Formato JSON:
{{
    "summary": "...",
    "metrics": {{...}},
    "red_flags": [...],
    "recommendation": "..."
}}"""
```

---

### ❌ Problema: Alucinaciones (información inventada)

**Síntomas:** LLM inventa hechos, citas inexistentes, datos falsos

**Soluciones:**

**1. Few-Shot Examples**
```python
prompt = """Responde SOLO basándote en el contexto.

Ejemplo:
Contexto: "La empresa facturó $100M en 2023"
Pregunta: "¿Cuánto facturó en 2024?"
Respuesta: "No tengo información sobre 2024"  # ← No inventa

Contexto: {context}
Pregunta: {question}
Respuesta:"""
```

**2. Grounding explícito**
```python
prompt = f"""Contexto: {retrieved_docs}

REGLAS ESTRICTAS:
- SOLO usa información del contexto
- Si no sabes, di "No tengo esa información"
- No extrapoles ni adivines
- Cita secciones del contexto

Pregunta: {question}"""
```

**3. Verificación post-generación**
```python
def verify_response(response, context):
    verification_prompt = f"""
    Contexto original: {context}
    Respuesta generada: {response}
    
    ¿La respuesta está completamente fundamentada en el contexto?
    Responde SI o NO y explica.
    """
    
    check = llm.invoke(verification_prompt)
    if "NO" in check:
        return "No puedo responder esa pregunta con información disponible"
    return response
```

---

## 5. RAG Específico

### ❌ Problema: Retrieval devuelve documentos irrelevantes

**Causas:**
- Embeddings de mala calidad
- Chunks mal hechos
- Metadata faltante

**Soluciones:**

**1. Hybrid Search**
```python
# No usar solo vectorial
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

**2. Chunking con overlap**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,  # ⚠️ IMPORTANTE
    separators=["\n\n", "\n", ". "]
)
```

**3. Reranking**
```python
from langchain.retrievers.document_compressors import CohereRerank

reranker = CohereRerank(top_n=3)
docs = reranker.compress_documents(retrieved_docs, query)
```

---

### ❌ Problema: "Context length exceeded"

**Síntomas:**
```
openai.error.InvalidRequestError: This model's maximum context length is 4096 tokens
```

**Soluciones:**

**1. Limitar documentos recuperados**
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Solo top 3, no 10
)
```

**2. Summarize antes de pasar**
```python
def summarize_long_context(docs):
    if count_tokens(docs) > 3000:
        summary_prompt = f"Summarize: {docs}"
        return llm.invoke(summary_prompt)
    return docs
```

**3. Usar modelo con  mayor contexto**
```python
llm = ChatOpenAI(model="gpt-4-turbo", max_tokens=128000)
```

---

## 6. Multi-Agente

### ❌ Problema: Agentes en loop infinito

**Síntomas:** Sistema nunca termina

**Solución:**
```python
# SIEMPRE tener safety net
MAX_ITERATIONS = 10
iteration = 0

while not done and iteration < MAX_ITERATIONS:
    result = agent.step()
    iteration += 1
    
if iteration >= MAX_ITERATIONS:
    logger.warning("Max iterations reached - forcing stop")
    return "Unable to complete task"
```

---

### ❌ Problema: Agentes no colaboran correctamente

**Causa:** Comunicación pobre

**Solución:**
```python
# Protocolo de mensaje estructurado
class AgentMessage:
    sender: str
    receiver: str
    message_type: str  # "request", "response", "notify"
    content: Dict
    requires_response: bool

# Validación
def send_message(msg: AgentMessage):
    if msg.requires_response:
        response = wait_for_response(msg.id, timeout=30)
        if not response:
            raise TimeoutError(f"No response from {msg.receiver}")
```

---

## 7. Costos

### ❌ Problema: Costos inesperadamente altos

**Síntomas:** $1000+ en una semana

**Causas:**
- Usar GPT-4 para todo
- No caching
- Loops excesivos
- Context innecesariamente grande

**Soluciones:**

**1. Caching**
```python
from langchain.cache import RedisCache
set_llm_cache(RedisCache())

# 30-40% reducción inmediata
```

**2. Model Routing**
```python
def get_model(complexity_score):
    if complexity_score < 0.3:
        return "gpt-4o-mini"  # $0.15/1M tokens
    return "gpt-4o"  # $2.50/1M tokens
```

**3. Monitorear**
```python
def track_cost(prompt, response):
    tokens = count_tokens(prompt) + count_tokens(response)
    cost = calculate_cost(tokens, model)
    
    db.insert({"timestamp": now(), "cost": cost, "tokens": tokens})
    
    if cost > daily_budget:
        alert("Budget exceeded!")
```

---

## 8. Testing

### ❌ Problema: Tests no determinísticos

**Causa:** LLM responses varían

**Solución:**
```python
# Mock el LLM en tests
from unittest.mock import Mock

def test_agent():
    llm_mock = Mock()
    llm_mock.invoke.return_value = "Expected output"
    
    agent = MyAgent(llm=llm_mock)
    result = agent.process("input")
    
    assert result == "Expected output"
```

---

## 9. Deployment

### ❌ Problema: Funciona en local pero falla en producción

**Causas:**
- Variables de entorno no configuradas
- Paths relativos
- Dependencias faltantes

**Checklist:**
```dockerfile
# Dockerfile completo
FROM python:3.11-slim

# Deps del sistema
RUN apt-get update && apt-get install -y gcc

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . /app
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "main.py"]
```

---

## 10. Debugging

### ❌ Problema: No sé qué está fallando

**Solución: Logging Estructurado**
```python
import structlog

logger = structlog.get_logger()

logger.info("agent_start", agent_id="analyst", task="analyze_stock")
logger.error("retrieval_failed", 
    query=query,
    docs_found=0,
    error=str(e)
)

# LangSmith para tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Ver traces en https://smith.langchain.com
```

---

## 📋 Quick Debug Checklist

Cuando algo falla:

1. [ ] ¿Variables de entorno cargadas? (`print(os.getenv("KEY"))`)
2. [ ] ¿API key válida? (Test con curl)
3. [ ] ¿Imports correctos? (`python -c "import X"`)
4. [ ] ¿Logs muestran algo? (Activar DEBUG level)
5. [ ] ¿Funciona el ejemplo más simple? (Hello World test)
6. [ ] ¿Versiones compatibles? (`pip list | grep langchain`)
7. [ ] ¿Network/firewall issues? (`curl https://api.openai.com`)
8. [ ] ¿Rate limits? (Check OpenAI dashboard)
9. [ ] ¿Context size? (Count tokens)
10. [ ] ¿Timeout muy corto? (Increase timeout)

---

## 🆘 Getting Help

### Antes de preguntar:

1. **Search first:** GitHub Issues, Stack Overflow
2. **Minimal reproduction:** Code que reproduce el problema
3. **Version info:** Python, framework versions
4. **Error completo:** Full traceback

### Dónde preguntar:

- **LangChain:** [Discord](https://discord.gg/langchain)
- **CrewAI:** [GitHub Issues](https://github.com/joaomdmoura/crewAI/issues)
- **AutoGen:** [GitHub Discussions](https://github.com/microsoft/autogen/discussions)
- **Stack Overflow:** Tag `langchain` / `openai`

---

**Pro Tip:** El 80% de los problemas se resuelven con:
1. Verificar variables de entorno
2. Actualizar librerías
3. Leer error messages completos
4. Google el error exacto

**Recuerda:** Si encuentras un bug en el framework, report it! Contribuir a open source beneficia a todos.
