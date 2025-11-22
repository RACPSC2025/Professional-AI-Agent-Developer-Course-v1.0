# Módulo 6: IA Confiable y Evaluación (Trustworthy AI)

## 🎯 Objetivos del Módulo
Es fácil hacer una demo que funcione el 80% de las veces. Lo difícil es llegar al 99%. En este módulo, aprenderás a medir la calidad de tus agentes, protegerlos contra ataques y asegurar que no filtren datos sensibles.

## 📚 Conceptos Clave

### 1. Evaluación (Evals)
-   **LLM-as-a-Judge:** Usar un modelo potente (GPT-4) para evaluar las respuestas de un modelo más pequeño.
-   **Métricas RAG:** Context Recall, Context Precision, Faithfulness (¿La respuesta se basa en el contexto o alucina?).
-   **Herramientas:** Ragas, DeepEval, LangSmith.

### 2. Guardrails (Barandillas)
-   Capas de seguridad que interceptan la entrada del usuario o la salida del agente.
-   **NeMo Guardrails:** Definir flujos permitidos y bloqueados.
-   **Guardrails AI:** Validadores programáticos (no PII, no toxicidad, no jailbreak).

### 3. Adversarial Testing (Red Teaming)
-   **Prompt Injection:** Intentos de manipular el sistema prompt.
-   **Jailbreak:** Evadir restricciones de seguridad.
-   **Data Leakage:** Extraer información sensible del prompt.
-   **Automated Red Team:** Usar LLMs para generar ataques automáticamente.

### 4. Bias y Fairness
-   Detectar sesgos en respuestas (género, raza, edad).
-   Métricas de equidad en sistemas de clasificación.
-   Debiasing techniques.

## 💻 Snippet de Código: Evaluación con Ragas

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Dataset de prueba
data = {
    'question': ['¿Cómo reseteo mi password?'],
    'answer': ['Ve a configuración y pulsa reset.'],
    'contexts': [['Para resetear password, ir a settings...']]
}

# Evaluar
results = evaluate(
    dataset=data,
    metrics=[faithfulness, answer_relevancy, context_recall]
)

print(results)
# {'faithfulness': 0.92, 'answer_relevancy': 0.85, 'context_recall': 0.78}
```

## 🛠️ Proyectos Prácticos

### 🟢 Nivel Básico: Sistema de Evaluación RAG
**Archivo:** `01_rag_evaluation_system.py`
-   **Concepto:** Evaluar calidad de respuestas RAG automáticamente.
-   **Framework:** LangChain + Ragas
-   **Caso de uso:** CI/CD pipeline que valida calidad antes de deploy.

### 🟡 Nivel Intermedio: Implementación de Guardrails
**Archivo:** `02_guardrails_implementation.py`
-   **Concepto:** Validadores de entrada/salida para protección.
-   **Framework:** Guardrails AI + LangChain
-   **Caso de uso:** Chatbot corporativo que bloquea información sensible.

### 🔴 Nivel Avanzado: Framework de Red Teaming
**Archivo:** `03_advanced_redteam_framework.py`
-   **Concepto:** Sistema automatizado de adversarial testing.
-   **Framework:** LangChain con evaluadores personalizados
-   **Caso de uso:** Auditoría de seguridad de agentes empresariales.

## 🎓 Mejores Prácticas

1. **Evaluar continuamente:** No solo al principio, también en producción.
2. **Usar múltiples métricas:** No hay una métrica perfecta.
3. **Test adversarial:** Asume que usuarios maliciosos intentarán romper tu sistema.
4. **Logging completo:** Registra todas las interacciones para análisis post-mortem.
5. **Human-in-the-loop:** Para decisiones críticas, siempre tener revisión humana.
6. **Versioning de prompts:** Trackear cambios en prompts como código.

