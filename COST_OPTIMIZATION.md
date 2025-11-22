# Cost Optimization Guide - AI Agent Development

## 🎯 Overview

Esta guía proporciona estrategias comprobadas para optimizar costos en sistemas de AI Agents.

---

## 💰 Cost Breakdown por Modelo (2024)

### OpenAI Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context Window | Speed |
|-------|----------------------|------------------------|----------------|-------|
| **GPT-4o** | $2.50 | $10.00 | 128K | Fast |
| **GPT-4o-mini** | $0.15 | $0.60 | 128K | Very Fast |
| **GPT-4-turbo** | $10.00 | $30.00 | 128K | Medium |
| **GPT-3.5-turbo** | $0.50 | $1.50 | 16K | Very Fast |

### Anthropic Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|-------|----------------------|------------------------|----------------|
| **Claude 3.5 Sonnet** | $3.00 | $15.00 | 200K |
| **Claude 3 Haiku** | $0.25 | $1.25 | 200K |

### Google Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|-------|----------------------|------------------------|----------------|
| **Gemini 1.5 Pro** | $1.25 | $5.00 | 1M |
| **Gemini 1.5 Flash** | $0.075 | $0.30 | 1M |

---

## 📊 Cost Calculator

### Ejemplo: Chatbot con 1,000 usuarios

**Escenario Base:**
- 1,000 usuarios/día
- 5 mensajes promedio/usuario
- 500 tokens promedio/mensaje (input + output)
- Modelo: GPT-4o

**Cálculo:**
```python
daily_users = 1000
messages_per_user = 5
tokens_per_message = 500

total_daily_tokens = daily_users * messages_per_user * tokens_per_message
# = 2,500,000 tokens/día

# GPT-4o: ~$5/1M tokens promedio (input+output)
daily_cost = (total_daily_tokens / 1_000_000) * 5
# = $12.50/día

monthly_cost = daily_cost * 30
# = $375/mes
```

**Con Optimizaciones (ver abajo):**
```python
# 40% reducción por caching = de $375 → $225
# 50% reducción por model routing = de $225 → $112.50
# Total: ~$113/mes (70% de ahorro)
```

---

## 🔧 Estrategia 1: Caching

### Semantic Caching

**Concepto:** Cachear respuestas similares, no solo idénticas.

```python
from langchain.cache import RedisSemanticCache
from langchain.embeddings import OpenAIEmbeddings

# Setup
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.92  # Ajustar
))

# Uso (transparente)
llm = ChatOpenAI(model="gpt-4o-mini")

# Primera vez: hit API
response1 = llm.invoke("¿Qué es Python?")  # $$$

# Preguntas similares: cache hit
response2 = llm.invoke("Explica qué es Python")  # $0
response3 = llm.invoke("What is Python programming")  # $0
```

**Ahorro:** 30-40% reducción en costos

### Exact Match Caching

```python
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# Para queries repetidas exactas
llm.invoke("Translate 'hello'")  # Cache miss
llm.invoke("Translate 'hello'")  # Cache hit - $0
```

**Ahorro:** 10-20% adicional

### Cache TTL Strategy

```python
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

# TTL según tipo de contenido
cache = RedisCache(redis_client)

def cache_with_ttl(key, value, ttl_seconds):
    redis_client.setex(key, ttl_seconds, value)

# Datos estáticos: TTL largo
cache_with_ttl("definition_python", response, ttl_seconds=86400)  # 24h

# Noticias: TTL corto
cache_with_ttl("latest_news", response, ttl_seconds=3600)  # 1h
```

---

## 🔧 Estrategia 2: Model Routing

### Intelligent Model Selection

```python
def classify_complexity(query: str) -> float:
    """Clasificar complejidad de 0-1"""
    complexity_prompt = f"""Rate query complexity 0-1:
    - Simple fact/definition: 0.1
    - Multi-step: 0.5
    - Complex reasoning: 0.9
    
    Query: {query}
    Score:"""
    
    # Usar modelo barato para clasificación
    classifier = ChatOpenAI(model="gpt-4o-mini")
    score = float(classifier.invoke(complexity_prompt).content)
    return score

def route_to_model(query: str) -> ChatOpenAI:
    """Routing basado en complejidad"""
    complexity = classify_complexity(query)
    
    if complexity < 0.3:
        return ChatOpenAI(model="gpt-4o-mini")  # $0.15/1M in
    elif complexity < 0.7:
        return ChatOpenAI(model="gpt-4o")  # $2.50/1M in
    else:
        return ChatOpenAI(model="gpt-4-turbo")  # $10/1M in

# Uso
query = "What is 2+2?"
llm = route_to_model(query)  # → gpt-4o-mini
response = llm.invoke(query)
```

**Ahorro:** 50-60% en queries simples

### Cascade Pattern

```python
async def cascade_query(query: str):
    """Intentar modelos baratos primero"""
    models = [
        ("gpt-4o-mini", 0.15),
        ("gpt-4o", 2.50),
        ("gpt-4-turbo", 10.00)
    ]
    
    for model_name, cost in models:
        llm = ChatOpenAI(model=model_name, temperature=0)
        response = await llm.ainvoke(query)
        
        # Evaluar calidad
        if quality_check(response):
            return response
    
    # Último recurso
    return response
```

**Ahorro:** 40-50% promedio

---

## 🔧 Estrategia 3: Prompt Optimization

### Token Reduction

```python
# ❌ Verbose (850 tokens)
prompt = """You are a helpful, professional, and knowledgeable assistant 
specialized in providing detailed, comprehensive, and accurate answers to 
user questions. You should always be polite, respectful, and considerate 
of the user's needs. When providing answers... [etc]"""

# ✅ Concise (120 tokens)
prompt = """Expert assistant. Provide accurate, concise answers.

Question: {question}
Answer:"""

# Ahorro: 85% menos tokens
```

### Few-Shot vs Zero-Shot

```python
# Few-shot when necessary
few_shot = """Examples:
Input: "Good product" → Positive
Input: "Bad quality" → Negative

Input: {text} →"""  # 25 tokens

# Zero-shot cuando el modelo puede
zero_shot = "Classify sentiment: {text}"  # 5 tokens

# Use zero-shot first, few-shot if accuracy < threshold
```

### Template Compression

```python
# Crear templates reusables
SYSTEM_SHORT = "Expert Q&A bot."
SYSTEM_LONG = "You are an expert assistant..."  # Solo si necesario

# Reusar en múltiples llamadas
chain = {"system": SYSTEM_SHORT} | llm
```

**Ahorro:** 20-30% en prompt tokens

---

## 🔧 Estrategia 4: Batch Processing

### Batch API Calls

```python
# ❌ Individual (5 llamadas, alto overhead)
for item in items:
    result = llm.invoke(f"Process: {item}")

# ✅ Batch (1 llamada)
batch_prompt = f"""Process in batch:
{chr(10).join(f'{i+1}. {item}' for i, item in enumerate(items))}

Results (JSON):"""

results = llm.invoke(batch_prompt)
```

**Ahorro:** 40-60% menos overhead

### Async Parallel Processing

```python
import asyncio

async def process_in_parallel(items):
    """Reduce latency sin aumentar costo"""
    tasks = [llm.ainvoke(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results

# 10 items en 2 segundos vs 20 segundos
```

---

## 🔧 Estrategia 5: Output Length Control

### Max Tokens Limit

```python
# Limitar output innecesariamente largo
llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=150  # Short answers
)

# Para FAQ: max_tokens=100
# Para resúmenes: max_tokens=300
# Para artículos: max_tokens=1000
```

**Ahorro:** 30-50% en output tokens (más caros)

### Structured Outputs

```python
# ❌ Verbose
prompt = "Analyze this and explain everything..."
# Response: 500 tokens de explicación

# ✅ Structured
prompt = """Analyze and return JSON:
{
  "summary": "...",  # max 50 words
  "score": 0-10,
  "key_points": [...]  # max 3
}"""
# Response: 100 tokens estructurados
```

---

## 🔧 Estrategia 6: RAG Optimization

### Reduce Retrieved Documents

```python
# ❌ Mucho context
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
docs = retriever.get_relevant_documents(query)  # 10k tokens

# ✅ Optimize
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = reranker.rerank(docs)  # Top 3, 3k tokens

# 70% menos input tokens
```

### Chunk Size Optimization

```python
# Experimentar con tamaño óptimo
chunk_sizes = [200, 400, 600, 800]

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size)
    # Medir: accuracy vs tokens
    
# Encontrar sweet spot (generalmente 400-600)
```

**Ahorro:** 20-40% en costos de RAG

---

## 📊 Monitoring y Tracking

### Cost Tracking

```python
import structlog

logger = structlog.get_logger()

def track_llm_call(prompt, response, model):
    tokens_in = count_tokens(prompt)
    tokens_out = count_tokens(response)
    cost = calculate_cost(tokens_in, tokens_out, model)
    
    logger.info("llm_call",
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost
    )
    
    # Guardar en DB
    db.insert({
        "timestamp": now(),
        "model": model,
        "cost": cost,
        "tokens": tokens_in + tokens_out
    })

# Análisis mensual
monthly_cost = db.query("SELECT SUM(cost) FROM llm_calls WHERE...")
```

### Budget Alerts

```python
def check_budget(user_id, cost):
    daily_limit = get_user_limit(user_id)
    today_spend = get_today_spend(user_id)
    
    if today_spend + cost > daily_limit:
        raise BudgetExceededError(
            f"Daily limit ${daily_limit} exceeded"
        )
    
    # Soft warning at 80%
    if today_spend + cost > daily_limit * 0.8:
        send_alert(f"⚠️ 80% of daily budget used")
```

---

## 💡 Quick Wins (ROI alto, esfuerzo bajo)

| Optimization | Effort | Savings | Implementation Time |
|--------------|--------|---------|---------------------|
| **Caching** | Low | 30-40% | 1-2 hours |
| **Model Routing** | Medium | 50-60% | 4-6 hours |
| **Prompt Optimization** | Low | 20-30% | 2-3 hours |
| **Output Length Limits** | Low | 30-50% | 30 min |
| **Batch Processing** | Medium | 40-60% | 2-4 hours |

---

## 📈 Case Study: Real Cost Reduction

### Before Optimization

```
Sistema: Customer support chatbot
Usuarios: 5,000/día
Mensajes: 25,000/día
Modelo: GPT-4-turbo
Tokens promedio: 800/mensaje

Costo mensual: $6,000
```

### After Optimization

```
Cambios implementados:
1. Semantic caching (40% hit rate)  
2. Model routing (60% queries → gpt-4o-mini)
3. Prompt optimization (30% menos tokens)
4. Output limits (150 max tokens)

Nuevo costo mensual: $1,800

Ahorro: $4,200/mes (70% reducción)
```

---

## 🎯 Cost Optimization Checklist

Antes de launch, verifica:

### Must-Have (Alta prioridad)
- [ ] Caching implementado
- [ ] Model routing configurado
- [ ] Output length limits
- [ ] Cost tracking habilitado
- [ ] Budget alerts configuradas

### Should-Have (Media prioridad)
- [  ] Prompt optimization
- [ ] Batch processing donde posible
- [ ] RAG chunk size optimizado
- [ ] Async processing
- [ ] Metrics dashboard

### Nice-to-Have (Baja prioridad)
- [ ] Cascade model selection
- [ ] Dynamic budget allocation
- [ ] A/B testing de prompts
- [ ] Cost anomaly detection

---

## 🚀 Roadmap de Optimización

### Mes 1: Fundamentos
- Implementar caching básico
- Añadir cost tracking
- Optimizar prompts obvios

### Mes 2: Model Routing
- Implementar clasificador de complejidad
- Routing a modelos apropiados
- Medir ahorros

### Mes 3: Avanzado
- Semantic caching
- Cascade patterns
- A/B testing continuo

---

**Pro Tip:** La optimización de costos es un proceso continuo, no one-time. Monitorea, mide, itera. Los modelos y precios cambian, tus optimizaciones también deben evolucionar.

**Recuerda:** El costo no es el único factor. Calidad y latencia también importan. Encuentra el balance correcto para tu caso de uso.
