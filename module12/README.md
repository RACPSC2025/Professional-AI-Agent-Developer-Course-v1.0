# Módulo 12: Protocolos de Agentes (The Agentic Web)

![Module 12 Banner](../images/module12_banner.png)

> "En 2025, la web no es de páginas, es de Agentes. Y necesitan un lenguaje común para hablar."

## 🎯 Objetivos del Módulo

Un agente aislado es útil. Un millón de agentes conectados es una economía. En este módulo, aprenderás los protocolos que hacen posible la **Agentic Web**.

**Lo que vas a dominar:**
1.  🔌 **MCP (Model Context Protocol):** El estándar para conectar agentes a *herramientas* (DBs, APIs).
2.  🤝 **A2A (Agent-to-Agent Protocol):** El estándar de Google para que *agentes* hablen entre sí.
3.  💰 **AP2 (Agent Payments Protocol):** Cómo los agentes se pagan mutuamente por servicios.

---

## 📚 Conceptos Clave (Nov 2025)

### 1. MCP vs A2A: La Diferencia Crítica

-   **MCP (Model Context Protocol):** Es el "USB" de los agentes. Conecta un cerebro (LLM) con un periférico (Herramienta).
    -   *Uso:* Tu agente leyendo un PDF o consultando SQL.
-   **A2A (Agent-to-Agent):** Es el "TCP/IP" de los agentes. Conecta un cerebro con otro cerebro.
    -   *Uso:* Tu agente de viajes negociando con el agente de la aerolínea.

### 2. El "Agentic Handshake"
Cuando dos agentes se encuentran en 2025, ocurre esto:
1.  **Discovery:** "¿Quién eres y qué puedes hacer?" (Vía DID - Decentralized ID).
2.  **Negotiation:** "¿Cuánto cobras por buscar este vuelo?" (Vía AP2).
3.  **Execution:** "Aquí tienes los parámetros, hazlo." (Vía A2A).

---

## 🌍 High Impact Social/Professional Example (Nov 2025)

> **Proyecto: "BabelNode" - El Traductor Universal de Agentes**
>
> Este ejemplo implementa un **Nodo Puente** que permite a un agente corporativo (IBM ACP) contratar servicios de un agente creativo (Google A2A) de forma transparente.

### El Problema
El ecosistema está fragmentado. Un agente de Supply Chain en IBM no "habla" el mismo idioma que un agente de Marketing en Google Cloud.

### La Solución
Un agente intermedio que traduce los protocolos en tiempo real, permitiendo colaboración cross-ecosystem.

```python
"""
Project: BabelNode
Protocol: A2A <-> ACP Bridge
Stack: Python, gRPC, Protobuf
"""
import asyncio
from protocols.google import A2A_Message
from protocols.ibm import ACP_Payload

class BabelBridge:
    def __init__(self):
        self.supported_protocols = ["A2A", "ACP", "MCP"]

    async def translate_intent(self, source_msg, target_proto):
        print(f"🔄 Translating from {source_msg.protocol} to {target_proto}...")
        
        # 1. Decodificar Intención (Semantic Parsing)
        intent = await self.parse_semantics(source_msg.content)
        # intent = {"action": "book_meeting", "time": "14:00"}
        
        # 2. Re-encapsular en Protocolo Destino
        if target_proto == "ACP":
            return ACP_Payload(
                performative="REQUEST",
                content=intent
            )
        elif target_proto == "A2A":
            return A2A_Message(
                type="TASK_REQUEST",
                payload=intent
            )

    async def broker_transaction(self, buyer_agent, seller_agent):
        # El Agente Comprador (Google) pide un servicio
        req = await buyer_agent.send("Necesito un logo")
        
        # Babel traduce para el Vendedor (IBM)
        translated_req = await self.translate_intent(req, "ACP")
        proposal = await seller_agent.receive(translated_req)
        
        # Babel traduce la oferta de vuelta
        final_offer = await self.translate_intent(proposal, "A2A")
        return final_offer

# Simulación
async def main():
    bridge = BabelBridge()
    print("🌐 BabelNode Active. Listening for cross-protocol traffic...")
    # ... traffic loop
```

**Impacto Profesional:**
- **Interoperabilidad Global:** Permite a las empresas elegir "el mejor agente para el trabajo" sin importar quién lo fabricó.
- **Mercado Líquido:** Crea una economía real donde agentes pequeños pueden vender servicios a grandes corporaciones.

---

## 🛠️ Proyectos Prácticos

### 🔌 Proyecto 1: Servidor MCP Universal
Crear un servidor que expone el sistema de archivos local a cualquier agente (Claude, OpenAI) de forma segura.

### 🤝 Proyecto 2: Negociación A2A Simple
Dos agentes (Comprador y Vendedor) que negocian el precio de un item simulado usando un protocolo de subasta.

### 🌐 Proyecto 3: BabelNode (Simulación)
El puente de traducción descrito arriba, conectando dos agentes con "idiomas" JSON incompatibles.

---

## 🚀 Próximos Pasos

➡️ **[Capstone Project: El Agente Maestro](../module13/README.md)**

<div align="center">

**[⬅️ Módulo Anterior](../module11/README.md)** | **[🏠 Inicio](../README.md)**

</div>

---

**Última actualización:** Noviembre 2025
**Stack:** MCP SDK, Google A2A, Python AsyncIO
**Conceptos:** Agent Interoperability, Decentralized AI
