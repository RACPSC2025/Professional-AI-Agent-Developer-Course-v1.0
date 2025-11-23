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

## 📧 Caso Práctico: Tu Agente de Gmail con MCP

Vamos a conectar tu Gmail real a un agente usando el servidor oficial de Google.

### Paso 1: Obtener Credenciales (Google Cloud)
Para que esto funcione, necesitas "llaves" de Google:
1.  Ve a **[Google Cloud Console](https://console.cloud.google.com/)** y crea un proyecto.
2.  Habilita la **Gmail API**.
3.  Crea credenciales **OAuth 2.0 (Desktop App)** y descarga el JSON.
4.  Obtén tu `Refresh Token` usando el [OAuth Playground](https://developers.google.com/oauthplayground) (Scopes: `https://mail.google.com/`).

---

## 🛠️ Configuración en VS Code (Copilot & Gemini)

VS Code es el editor estándar. A partir de 2025, soporta MCP nativamente.

1.  Crea una carpeta `.vscode` en la raíz de tu proyecto.
2.  Crea un archivo `mcp.json` dentro:

```json
{
  "mcpServers": {
    "gmail": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-gmail"],
      "env": {
        "GMAIL_CLIENT_ID": "TU_CLIENT_ID",
        "GMAIL_CLIENT_SECRET": "TU_CLIENT_SECRET",
        "GMAIL_REFRESH_TOKEN": "TU_REFRESH_TOKEN"
      }
    }
  }
}
```

3.  **Reinicia VS Code.**
4.  Abre **GitHub Copilot Chat** o **Gemini Code Assist**.
5.  Escribe: *"@gmail busca los correos no leídos de hoy"*.
    *   El editor te pedirá permiso para ejecutar el comando `npx`. Acepta y observa la magia.

---

## 💻 Configuración Avanzada: CLIs y Antigravity

Si eres fan de la terminal, así se configuran los CLIs oficiales:

#### 1. Gemini CLI (`gemini-cli`)
Archivo: `~/.gemini/settings.json`

```json
{
  "mcpServers": {
    "gmail": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-gmail"],
      "env": {
        "GMAIL_CLIENT_ID": "TU_CLIENT_ID",
        "GMAIL_CLIENT_SECRET": "TU_CLIENT_SECRET",
        "GMAIL_REFRESH_TOKEN": "TU_REFRESH_TOKEN"
      }
    }
  }
}
```

#### 2. Qwen CLI (`qwen-code`)
Archivo: `~/.qwen/settings.json`

```json
{
  "mcpServers": {
    "gmail": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-gmail"],
      "env": {
        "GMAIL_CLIENT_ID": "TU_CLIENT_ID",
        "GMAIL_CLIENT_SECRET": "TU_CLIENT_SECRET",
        "GMAIL_REFRESH_TOKEN": "TU_REFRESH_TOKEN"
      }
    }
  }
}
```

### 🤖 Antigravity (Yo) y Tú
¿Cómo me usas a mí (**Antigravity**) con estas herramientas?

¡Es automático! Al estar yo integrado en tu entorno de desarrollo (VS Code / Cursor):
1.  Sigue los pasos de **"Configuración en VS Code"** de arriba.
2.  Una vez creado el archivo `.vscode/mcp.json`, yo detecto automáticamente las herramientas disponibles.
3.  Solo dime: *"Antigravity, usa Gmail para enviarle este código a mi manager"* y yo me encargo del resto.

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

---

---

## 🚀 Proyecto Avanzado: Tu Propio Agente MCP Full-Stack

¿Quieres ir más allá de la configuración? Vamos a construir un sistema completo: **Servidor Propio + Cliente CLI + Interfaz Web**.

Hemos creado una carpeta `project_mcp_agent` con 3 archivos clave:

### 1. El Cerebro (Servidor MCP)
Archivo: `module12/project_mcp_agent/server.py`
Un servidor seguro hecho con Python que expone herramientas críticas (como encriptación).

### 2. La Terminal (Cliente CLI)
Archivo: `module12/project_mcp_agent/client_cli.py`
Para probar tus herramientas rápidamente desde la consola.
> **Ejecutar:** `python client_cli.py`

### 3. La Interfaz (Streamlit UI)
Archivo: `module12/project_mcp_agent/app.py`
Una web completa donde eliges tu LLM (GPT-4, Claude), pones tu API Key y chateas con tus herramientas.
> **Ejecutar:** `streamlit run app.py`

### 🌐 Escalabilidad Infinita
Lo mágico de esto es que **puedes conectar CUALQUIER servidor MCP** a este mismo cliente. No solo el tuyo, sino los cientos disponibles en la comunidad:

*   **[Official MCP Servers Repo](https://github.com/modelcontextprotocol/servers):** Conecta Google Drive, Slack, Postgres, GitHub, etc.
*   **[Glama MCP Registry](https://glama.ai/mcp/servers):** Un directorio visual de servidores comunitarios.

Tu agente puede empezar enviando correos y terminar gestionando toda tu infraestructura en la nube, solo añadiendo más servidores a la lista.

---

➡️ **[Capstone Project: El Agente Maestro](../module13/README.md)**

<div align="center">

**[⬅️ Módulo Anterior](../module11/README.md)** | **[🏠 Inicio](../README.md)**

</div>

---

**Última actualización:** Noviembre 2025
**Stack:** MCP SDK, Google A2A, Python AsyncIO
**Conceptos:** Agent Interoperability, Decentralized AI
