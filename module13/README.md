# Módulo 13: Testing de Agentes

## 🎯 Objetivos del Módulo

Aprender a testear AI Agents de forma profesional, desde unit tests hasta pipelines CI/CD completos.

## 📚 Conceptos Clave

### 1. Unit Testing

**Concepto:** Testear componentes individuales de agentes de forma aislada

**Desafío Testing único de LLMs:**
- Outputs no determinísticos
- Necesidad de mocking
- Evaluación de calidad (no solo igualdad exacta)

**Soluciones:**
- Mock LLM calls para tests determinísticos  
- LLM-as-a-Judge para evaluar calidad
- Similarity metrics
- Gold standard datasets

### 2. Integration Testing

**Concepto:** Testear flujos completos de múltiples agentes

**Qué testear:**
- Comunicación entre agentes
- Manejo de errores en cascada
- Performance end-to-end
- Estado compartido correctamente

### 3. CI/CD para Agentes

**Diferencia con CI/CD tradicional:**
- Evaluation metrics vs traditional assertions
- LLM-powered tests
- Dataset versioning
- Prompt versioning

## 🛠️ Proyectos Prácticos

### 🟢 Nivel Básico: Unit Testing Agents
**Archivo:** `01_unit_testing_agents.py`
- **Framework:** pytest con mocking
- **Concepto:** Tests determinísticos con LLM mocks
- **Coverage:** Funciones, tools, prompts

### 🟡 Nivel Intermedio: Integration Testing Multi-Agent
**Archivo:** `02_integration_testing_multiagent.py`
- **Framework:** pytest con fixtures
- **Concepto:** Test workflows completos
- **Caso de uso:** Sistema multi-agente research → analysis → report

### 🔴 Nivel Avanzado: CI/CD Pipeline
**Archivo:** `03_cicd_pipeline_agents.py`
- **Framework:** GitHub Actions + pytest
- **Concepto:** Automated testing en cada commit/PR
- **Includes:** Regression tests, performance benchmarks

## 🎓 Best Practices

### Testing Pyramid para Agentes

```
        /\
       /  \  E2E Tests (5%)
      /    \  Integration Tests (25%)
     /      \ Unit Tests (70%)
    /________\
```

### Métricas Clave

- **Test Coverage:** >80% de funciones
- **Response Quality:** Score >0.8 en evaluaciones
- **Latency:** p95 < threshold
- **Cost:** $ per test run

### Golden Rules

1. **Mock LLMs en unit tests** (rápido,  barato, determinístico)
2. **Use real LLMs en integration** (catch real issues)
3. **Version prompts** (git track cambios)
4. **Maintain gold datasets** (regression detection)
5. **Automate everything** (CI/CD esencial)

## 📊 Test Example Pattern

```python
def test_agent_response():
    # Arrange
    llm_mock = Mock()
    llm_mock.invoke.return_value = "Expected output"
    agent = MyAgent(llm=llm_mock)
    
    # Act
    result = agent.process("test input")
    
    # Assert
    assert result == "Expected output"
    llm_mock.invoke.assert_called_once()
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agents --cov-report=html

# Specific test
pytest tests/test_agent.py::test_specific_function
```

## 📚 Recursos

- [pytest Documentation](https://docs.pytest.org)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
