# Best Practices - AI Agent Development

## 🎯 Consolidated Best Practices

Esta guía consolida las mejores prácticas de todos los módulos del curso en un solo documento de referencia.

---

## 1. Prompting y Diseño de Agentes

### ✅ DO's

**Ser Específico y Claro**
```python
# ❌ Mal
prompt = "Eres un asistente"

# ✅ Bien
prompt = """Eres un analista financiero senior especializado en análisis de riesgo.
Tu rol es evaluar inversiones y proporcionar recomendaciones basadas en:
1. Análisis fundamental
2. Ratios financieros
3. Tendencias del mercado

Siempre incluye:
- Nivel de riesgo (bajo/medio/alto)
- Justificación basada en datos
- Alternativas si aplica"""
```

**Usar Few-Shot Examples**
```python
prompt = """Clasifica el sentimiento de estas reseñas:

Ejemplos:
Input: "El producto es excelente, superó mis expectativas"
Output: {"sentiment": "positive", "confidence": 0.95}

Input: "No funciona como esperaba, decepcionante"
Output: {"sentiment": "negative", "confidence": 0.90}

Ahora clasifica:
Input: {user_input}
Output:"""
```

**Estructurar Outputs**
```python
# Siempre pedir formato estructurado (JSON, XML, etc.)
prompt = """Analiza este texto y responde en JSON:
{
  "summary": "...",
  "key_points": [...],
  "sentiment": "positive|negative|neutral"
}"""
```

### ❌ DON'Ts

- ❌ Prompts vagos o ambiguos
- ❌ Asumir conocimiento del contexto
- ❌ Pedir múltiples cosas en un solo prompt
- ❌ Olvidar especificar el formato de salida

---

## 2. RAG (Retrieval-Augmented Generation)

### ✅ DO's

**Usar Hybrid Search Siempre**
```python
# Combinar vectorial (semántica) + BM25 (keywords)
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Ajustar según caso
)
```

**Implementar Reranking**
```python
# Reordenar resultados con cross-encoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank(model="rerank-english-v2.0", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble
)
```

**Chunking Inteligente**
```python
# Usar semantic chunking, no solo caracteres
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Experimentar 200-800
    chunk_overlap=50,  # 10% del chunk_size
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Añadir Metadata Rica**
```python
# Metadata ayuda enormemente en filtrado
doc = Document(
    page_content=content,
    metadata={
        "source": "docs/api.md",
        "date": "2024-01-15",
        "category": "authentication",
        "version": "2.0"
    }
)
```

### ❌ DON'Ts

- ❌ Usar solo búsqueda vectorial
- ❌ Chunks demasiado grandes (>1000 tokens) o pequeños (<100 tokens)
- ❌ Ignorar metadata
- ❌ No evaluar calidad de retrieval (context precision/recall)

---

## 3. Multi-Agente

### ✅ DO's

**Roles Claros y Específicos**
```python
# Cada agente debe tener rol bien definido
analyst = Agent(
    role='Financial Analyst',
    goal='Analyze company financials and identify risks',
    backstory='10 years experience in equity research...',
    tools=[calculator, web_search]
)
```

**Comunicación Explícita**
```python
# Define protocolos de comunicación claros
class Message:
    sender: str
    receiver: str
    message_type: MessageType  # REQUEST, RESPONSE, NOTIFY
    content: Dict
    conversation_id: str
```

**Evitar Loops Infinitos**
```python
# Siempre tener condiciones de salida
workflow.add_conditional_edges(
    "agent_a",
    should_continue,
    {
        "continue": "agent_b",
        "end": END  # ⚠️ CRÍTICO
    }
)

MAX_ITERATIONS = 10  # Safety net
```

### ❌ DON'Ts

- ❌ Agentes con responsabilidades solapadas
- ❌ Comunicación no estructurada
- ❌ No limitar iteraciones
- ❌ Olvidar manejo de conflictos

---

## 4. Seguridad y Trustworthy AI

### ✅ DO's

**Implementar Guardrails**
```python
# Input validation
def validate_input(user_input: str) -> bool:
    # Check for injection attempts
    if any(pattern in user_input.lower() for pattern in [
        "ignore previous", "system:", "admin", "<script>"
    ]):
        return False
    return True

# Output filtering
def filter_output(response: str) -> str:
    # Remove any PII, secrets
    response = redact_emails(response)
    response = redact_phone_numbers(response)
    return response
```

**Evaluar Continuamente**
```python
# LLM-as-a-Judge
evaluator_prompt = """Evalúa si esta respuesta:
1. Responde la pregunta completamente
2. Es factual (no alucina)
3. Es apropiada (no ofensiva)

Respuesta: {response}
Score (0-10):"""
```

**Red Teaming Regular**
```python
# Automated adversarial testing
attack_types = [
    "prompt_injection",
    "jailbreak",
    "data_leakage"
]

for attack_type in attack_types:
    results = red_team_framework.test(agent, attack_type)
    assert results.security_score > 0.8
```

### ❌ DON'Ts

- ❌ Exponer system prompts al usuario
- ❌ Hardcodear información sensible
- ❌ Confiar ciegamente en el output del LLM
- ❌ No testear contra ataques

---

## 5. Performance y Costos

### ✅ DO's

**Caching Agresivo**
```python
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache

# 30-40% de queries se repiten
set_llm_cache(RedisCache(redis_client))
```

**Model Routing Inteligente**
```python
def route_to_model(query_complexity: float):
    if query_complexity < 0.3:
        return "gpt-4o-mini"  # Cheap
    elif query_complexity < 0.7:
        return "gpt-4o"  # Balanced
    else:
        return "gpt-4"  # Expensive but capable
```

**Batch Processing**
```python
# Procesar múltiples queries en batch
responses = llm.batch([query1, query2, query3])  # 1 API call
```

**Streaming para UX**
```python
# Mostrar tokens mientras se generan
for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)
```

### ❌ DON'Ts

- ❌ Usar GPT-4 para todo
- ❌ No implementar caching
- ❌ Llamadas síncronas cuando podrían ser paralelas
- ❌ No monitorear costos

---

## 6. Testing y Calidad

### ✅ DO's

**Unit Tests con Mocking**
```python
import pytest
from unittest.mock import Mock

def test_agent_response():
    llm_mock = Mock()
    llm_mock.invoke.return_value = "Expected output"
    
    agent = MyAgent(llm=llm_mock)
    result = agent.process("test input")
    
    assert result == "Expected output"
    llm_mock.invoke.assert_called_once()
```

**Integration Tests**
```python
def test_multi_agent_workflow():
    manager = AgentManager()
    analyst = AnalystAgent()
    writer = WriterAgent()
    
    result = manager.orchestrate(
        task="Analyze AAPL",
        agents=[analyst, writer]
    )
    
    assert "analysis" in result
    assert "report" in result
```

**Regression Testing**
```python
# Mantener dataset gold standard
GOLD_STANDARD = [
    {"input": "...", "expected": "...", "min_score": 0.8},
    # ... más casos
]

def test_regression():
    for case in GOLD_STANDARD:
        result = agent.process(case["input"])
        score = similarity(result, case["expected"])
        assert score >= case["min_score"]
```

### ❌ DON'Ts

- ❌ Solo testar manualmente
- ❌ No usar mocks (caro y lento)
- ❌ Ignorar regression tests
- ❌ No versionar prompts

---

## 7. Deployment y Production

### ✅ DO's

**Containerization**
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["python", "app.py"]
```

**Secrets Management**
```python
# ❌ MAL
API_KEY = "sk-abc123..."

# ✅ BIEN
import os
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
```

**Monitoring y Alertas**
```python
# Log todo con structured logging
import structlog

logger = structlog.get_logger()

logger.info("agent_invoked", 
    agent_id="analyst_01",
    tokens=500,
    latency_ms=1250,
    cost_usd=0.002
)
```

### ❌ DON'Ts

- ❌ Deployar sin health checks
- ❌ No tener rollback plan
- ❌ Hardcodear secrets
- ❌ No monitorear en producción

---

## 8. Observability

### ✅ DO's

**Tracing Completo**
```python
# LangSmith integration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production"

# Cada llamada se registra automáticamente
```

**Métricas Clave**
```python
metrics_to_track = {
    "latency_p50": latency_p50,
    "latency_p95": latency_p95,
    "tokens_per_request": avg_tokens,
    "cost_per_request": avg_cost,
    "error_rate": error_rate,
    "user_satisfaction": satisfaction_score
}
```

### ❌ DON'Ts

- ❌ No trackear métricas
- ❌ Log solo a stdout
- ❌ No tener dashboards
- ❌ Ignorar anomalías

---

## 📋 Production Readiness Checklist

Antes de deployar a producción, verifica:

### Code Quality
- [ ] Tests con >80% coverage
- [ ] Código sigue PEP 8
- [ ] Type hints en funciones públicas
- [ ] Docstrings completos
- [ ] No hardcoded secrets

### Performance
- [ ] Caching implementado
- [ ] Model routing configurado
- [ ] Timeouts apropiados
- [ ] Rate limiting en place

### Security
- [ ] Guardrails implementados
- [ ] Input validation
- [ ] Output filtering
- [ ] Red teaming completado
- [ ] Secrets en vault

### Observability
- [ ] Logging estructurado
- [ ] Tracing habilitado
- [ ] Métricas configuradas
- [ ] Alertas definidas
- [ ] Dashboard creado

### Deployment
- [ ] Dockerizado
- [ ] CI/CD pipeline
- [ ] Health checks
- [ ] Rollback plan
- [ ] Load testing completado

---

## 🎓 Anti-Patterns Comunes

### ❌ God Agent
```python
# NO hacer esto
mega_agent = Agent(
    role="Do everything",
    tasks=["analyze", "write", "code", "design", ...]
)
```
**Solución:** Divide en agentes especializados

### ❌ Prompt Bloat
```python
# NO hacer esto
prompt = """You are a helpful assistant that...
[5000 palabras de instrucciones]
"""
```
**Solución:** System prompt conciso, ejemplos en el context

### ❌ No Handling Failures
```python
# NO hacer esto
result = llm.invoke(prompt)  # ¿Y si falla?
```
**Solución:** Try-except, retries, fallbacks

### ❌ Ignoring Context Limits
```python
# NO hacer esto
huge_context = "\n".join(all_documents)  # 100k tokens
result = llm.invoke(huge_context + prompt)  # 💥
```
**Solución:** Chunking, summarization, retrieval

---

## 🚀 Quick Wins

### Fácil de Implementar, Alto Impacto

1. **Añadir Caching** → 30-40% reducción de costos
2. **Implementar Reranking** → 20-30% mejora en RAG accuracy
3. **Structured Outputs (JSON)** → 50% menos errores de parsing
4. **Model Routing** → 50-60% reducción de costos
5. **Streaming** → Perceived latency -80%

---

**Recuerda:** Estas best practices son guidelines, no reglas absolutas. Adapta según tu caso de uso específico. Lo que funciona para un chatbot simple no es igual para un sistema multi-agente enterprise.

**Pro Tip:** Implementa gradually. No intentes aplicar todo de una vez. Prioriza según impacto vs esfuerzo.
