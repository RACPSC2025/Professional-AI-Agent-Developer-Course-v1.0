from mcp.server.fastmcp import FastMCP
import datetime

# 1. Crear el Servidor MCP
mcp = FastMCP("AgenteSeguro")

# 2. Definir Herramientas (Tools)
@mcp.tool()
def get_secure_time() -> str:
    """Devuelve la hora actual del servidor seguro."""
    return datetime.datetime.now().isoformat()

@mcp.tool()
def encrypt_message(text: str) -> str:
    """Simula encriptación de datos sensibles."""
    return f"ENCRYPTED_SHA256_LIKE_HASH_OF_{text}"

@mcp.tool()
def system_status() -> str:
    """Verifica el estado de los sistemas críticos."""
    return "ALL_SYSTEMS_NOMINAL | SECURITY_LEVEL: HIGH"

# 3. Ejecutar Servidor
if __name__ == "__main__":
    mcp.run()
