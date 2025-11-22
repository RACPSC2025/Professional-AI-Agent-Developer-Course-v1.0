# Módulo 11: Protocolos de Agentes (MCP, A2A)

## 🎯 Objetivos del Módulo
El futuro no es un solo agente gigante, sino millones de agentes pequeños hablando entre sí. En este módulo, aprenderás los estándares que permitirán que tu agente de ventas hable con el agente de inventario de otra empresa.

## 📚 Conceptos Clave

### 1. Model Context Protocol (MCP)
-   El estándar abierto propuesto por Anthropic (y otros) para estandarizar cómo los modelos acceden a datos externos.
# Módulo 11: Protocolos de Agentes (MCP, A2A)

## 🎯 Objetivos del Módulo
El futuro no es un solo agente gigante, sino millones de agentes pequeños hablando entre sí. En este módulo, aprenderás los estándares que permitirán que tu agente de ventas hable con el agente de inventario de otra empresa.

## 📚 Conceptos Clave

### 1. Model Context Protocol (MCP)
-   El estándar abierto propuesto por Anthropic (y otros) para estandarizar cómo los modelos acceden a datos externos.
-   Evita tener que escribir una integración específica para cada nueva herramienta.

### 2. Agent-to-Agent (A2A) Communication
-   ¿Cómo se saludan dos agentes? ¿Cómo negocian? ¿Cómo se transfieren tareas?
-   Formatos de mensaje estándar (JSON-LD, AgentSpeak conceptual).

## 🛠️ Proyectos Prácticos (Niveles de Dificultad)

### 🟢 Nivel Básico: Cliente MCP
-   **Concepto:** Consumo de Protocolos Estándar.
-   **Misión:** Conectar tu agente a un servidor MCP existente (ej. Google Drive MCP) para listar archivos sin escribir código de integración específico.

### 🟡 Nivel Intermedio: Servidor MCP Personalizado
-   **Concepto:** Exponer Datos.
-   **Misión:** Crear un servidor MCP que exponga una base de datos SQLite local.
-   **Resultado:** Ahora CUALQUIER agente (Claude, tu agente LangChain, etc.) puede consultar tu DB simplemente conectándose al servidor.

### 🔴 Nivel Avanzado: Mercado de Agentes (A2A)
-   **Concepto:** Negociación entre Agentes.
-   **Misión:**
    -   **Agente Comprador:** Quiere reservar un vuelo por menos de $500.
    -   **Agentes Vendedores (x3):** Aerolíneas con precios dinámicos.
    -   Los agentes negocian en un protocolo estandarizado hasta cerrar el trato.

## 💻 Snippet: Estructura de Mensaje A2A (Conceptual)

```json
{
  "sender": "agent_sales_01",
  "receiver": "agent_inventory_05",
  "performative": "REQUEST",
  "content": {
    "action": "check_stock",
    "item_id": "SKU-12345",
    "quantity": 10
  },
  "protocol": "fipa-request",
  "language": "json"
}
```
