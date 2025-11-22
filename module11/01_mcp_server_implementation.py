"""
🟢 NIVEL CONCEPTUAL: MODEL CONTEXT PROTOCOL (MCP) SERVER
-------------------------------------------------------
Este ejemplo demuestra conceptualmente cómo implementar un MCP Server.
Caso de Uso: Exponer herramientas personalizadas via protocolo estándar.

Conceptos Clave:
- MCP: Protocolo de interoperabilidad para agentes
- Server implementation: Exposición de recursos y tools
- Client-server architecture: Separación de concerns

IMPORTANTE: MCP es un protocolo nuevo de Anthropic (2024).
Para implementación real, instalar: pip install mcp

ESTRUCTURA CONCEPTUAL:
=====================

```python
from mcp.server import Server, Tool
from mcp.types import TextContent

# 1. CREAR SERVIDOR MCP
server = Server("mi-empresa-tools")

# 2. DEFINIR HERRAMIENTAS (TOOLS)
@server.tool()
async def get_customer_data(customer_id: str) -> str:
    '''
    Obtiene datos de un cliente desde CRM interno.
    
    Args:
        customer_id: ID único del cliente
    
    Returns:
        Datos del cliente en formato JSON
    '''
    # En producción, consultaría BD real
    return json.dumps({
        "id": customer_id,
        "name": "Acme Corp",
        "status": "active",
        "revenue": 150000
    })

@server.tool()
async def create_support_ticket(description: str, priority: str) -> str:
    '''
    Crea un ticket de soporte en sistema interno.
    
    Args:
        description: Descripción del problema
        priority: low, medium, high
    
    Returns:
        ID del ticket creado
    '''
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    return f"Ticket {ticket_id} creado con prioridad {priority}"

# 3. DEFINIR RECURSOS (RESOURCES)
@server.resource("company://policies/refund")
async def get_refund_policy():
    '''
    Política de reembolsos de la empresa.
    '''
    return TextContent(
        uri="company://policies/refund",
        mimeType="text/markdown",
        text='''
# Política de Reembolsos

1. Productos físicos: 30 días
2. Software: 14 días sin uso
3. Servicios: No aplican reembolsos
        '''
    )

# 4. EJECUTAR SERVIDOR
if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run_server(server)
```

USO DESDE UN CLIENTE:
====================

```python
from mcp.client import Client

# Conectar a servidor MCP
client = Client()
await client.connect("stdio://mi-empresa-tools")

# Listar herramientas disponibles
tools = await client.list_tools()
print(tools)  # ['get_customer_data', 'create_support_ticket']

# Invocar herramienta
result = await client.call_tool(
    "get_customer_data",
    arguments={"customer_id": "CUST-123"}
)
print(result)

# Leer recurso
policy = await client.read_resource("company://policies/refund")
print(policy.text)
```

VENTAJAS DE MCP:
===============
✅ Interoperabilidad: Agentes de diferentes frameworks pueden usar mismo servidor
✅ Estandarización: Protocolo común vs. integraciones custom
✅ Desacoplamiento: Lógica de negocio separada del agente
✅ Seguridad: Control centralizado de acceso a datos sensibles

CASOS DE USO:
============
- Exponer APIs internas a agentes de forma segura
- Compartir herramientas entre equipos (MCP marketplace)
- Integrar sistemas legacy con agentes modernos
- Crear "Agent Operating System" (AOS)

COMPARACIÓN CON FUNCTION CALLING TRADICIONAL:
============================================
Traditional Function Calling:
  - Cada agente define sus propias funciones
  - Código duplicado si múltiples agentes necesitan misma función
  - Difícil versionar y actualizar

MCP:
  - Servidor centralizado de herramientas
  - Múltiples clientes (agentes) consumen mismas tools
  - Actualizar servidor actualiza todos los agentes

ARQUITECTURA:
============
┌─────────────┐
│ LangChain   │──┐
│ Agent       │  │
└─────────────┘  │
                 │    ┌──────────────────┐
┌─────────────┐  ├───▶│  MCP Server      │──▶ CRM Database
│ CrewAI      │  │    │  (Tools + Resources)
│ Agent       │  │    └──────────────────┘
└─────────────┘  │
                 │
┌─────────────┐  │
│ AutoGen     │──┘
│ Agent       │
└─────────────┘

INSTALACIÓN:
===========
pip install mcp
pip install anthropic  # Si usas Claude

RECURSOS:
========
- Docs: https://modelcontextprotocol.io/
- GitHub: https://github.com/anthropics/mcp
- Ejemplos: https://github.com/anthropics/mcp-servers
"""

print("""
="*70)
  🌐 MODEL CONTEXT PROTOCOL (MCP) - CONCEPTUAL OVERVIEW
="*70)

MCP es un protocolo estándar para interoperabilidad de agentes, creado por
Anthropic en 2024.

💡 ANALOGÍA:
  - HTTP es para web servers
  - MCP es para AI agent servers

📦 Este módulo es CONCEPTUAL. Para implementación real:
  1. Instalar: pip install mcp
  2. Estudiar docs: https://modelcontextprotocol.io/
  3. Ver ejemplos oficiales de Anthropic

🔧 EJEMPLO RÁPIDO DE USO:

# server.py
from mcp.server import Server

server = Server("my-tools")

@server.tool()
async def calculator(expression: str) -> float:
    return eval(expression)  # ⚠️ Inseguro, solo demo

# client.py (en tu agente)
from mcp.client import Client

client = await Client().connect("stdio://my-tools")
result = await client.call_tool("calculator", {"expression": "2 + 2"})
# result = 4.0

="*70)
    """)
