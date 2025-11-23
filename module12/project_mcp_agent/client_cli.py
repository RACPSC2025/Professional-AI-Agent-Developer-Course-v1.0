import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_cli():
    # Configuración de conexión al servidor local
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"], # Ejecuta nuestro propio server.py
        env=None
    )

    print("🔌 Conectando al Servidor MCP Seguro...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Inicializar
            await session.initialize()
            
            # 2. Listar Herramientas
            tools = await session.list_tools()
            print(f"\n🛠️  Herramientas Detectadas: {[t.name for t in tools.tools]}")
            
            # 3. Loop de interacción
            while True:
                user_input = input("\n👤 Tú (escribe 'salir' o nombre de herramienta): ")
                if user_input.lower() == 'salir': break
                
                # Ejemplo simple: Si el usuario escribe el nombre de una herramienta, la ejecutamos
                # En un agente real, un LLM decidiría esto.
                found = False
                for tool in tools.tools:
                    if tool.name == user_input:
                        print(f"🤖 Ejecutando {tool.name}...")
                        result = await session.call_tool(tool.name, arguments={})
                        print(f"📄 Resultado: {result.content[0].text}")
                        found = True
                
                if not found:
                    print("⚠️ Herramienta no reconocida o comando inválido.")

if __name__ == "__main__":
    asyncio.run(run_cli())
