"""
Módulo 11 - Ejemplo Intermedio: Cliente MCP Multi-Source
Framework: MCP (Model Context Protocol)
Caso de uso: Agente que conecta con Slack, GitHub y Google Drive vía MCP

Este ejemplo demuestra cómo un cliente MCP puede conectarse a múltiples
servidores para acceder a diferentes fuentes de datos de manera estandarizada.

Instalación:
pip install mcp anthropic
"""

import os
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Nota: Este es un ejemplo educativo. MCP está en desarrollo activo.
# La API puede cambiar. Consulta https://modelcontextprotocol.io


class MCPClient:
    """Cliente MCP que se conecta a múltiples servidores"""
    
    def __init__(self):
        self.connected_servers = []
        print("🔌 MCP Client inicializado")
    
    async def connect_to_server(self, server_config: Dict):
        """Conectar a un servidor MCP"""
        server_name = server_config["name"]
        print(f"\n📡 Conectando a servidor MCP: {server_name}...")
        
        # Simulación de conexión (en producción usarías el SDK real de MCP)
        await asyncio.sleep(0.5)
        
        self.connected_servers.append({
            "name": server_name,
            "type": server_config["type"],
            "capabilities": server_config.get("capabilities", [])
        })
        
        print(f"   ✅ Conectado a {server_name}")
        print(f"   Capacidades: {', '.join(server_config.get('capabilities', []))}")
    
    async def list_resources(self, server_name: str) -> List[Dict]:
        """Listar recursos disponibles en un servidor"""
        print(f"\n📋 Listando recursos de {server_name}...")
        
        # Simulación (en realidad llamaría al servidor MCP)
        simulated_resources = {
            "slack-mcp": [
                {"uri": "slack://channel/general", "name": "#general", "type": "channel"},
                {"uri": "slack://channel/tech", "name": "#tech", "type": "channel"},
                {"uri": "slack://dm/user123", "name": "DM con Alice", "type": "direct_message"},
            ],
            "github-mcp": [
                {"uri": "github://repo/myproject", "name": "my-project", "type": "repository"},
                {"uri": "github://issue/123", "name": "Issue #123", "type": "issue"}, 
                {"uri": "github://pr/45", "name": "PR #45", "type": "pull_request"},
            ],
            "gdrive-mcp": [
                {"uri": "gdrive://folder/docs", "name": "Documentos", "type": "folder"},
                {"uri": "gdrive://file/report.pdf", "name": "Report Q4.pdf", "type": "file"},
            ]
        }
        
        resources = simulated_resources.get(server_name, [])
        
        for resource in resources:
            print(f"   - {resource['name']} ({resource['type']})")
        
        return resources
    
    async def read_resource(self, uri: str) -> str:
        """Leer contenido de un recurso viaURI"""
        print(f"\n📖 Leyendo recurso: {uri}...")
        
        # Simulación de lectura
        await asyncio.sleep(0.3)
        
        # Datos simulados según el tipo de URI
        simulated_content = {
            "slack://channel/general": "Canal #general: Últimos mensajes sobre el lanzamiento del producto...",
            "github://repo/myproject": "README del proyecto my-project: Una aplicación de...",
            "github://issue/123": "Issue #123: Bug en el login - Los usuarios reportan...",
            "gdrive://file/report.pdf": "Report Q4: Resumen ejecutivo de resultados trimestrales...",
        }
        
        content = simulated_content.get(uri, f"Contenido de {uri}")
        print(f"   ✅ Contenido leído ({len(content)} caracteres)")
        
        return content
    
    async def query_cross_platform(self, query: str) -> List[Dict]:
        """Buscar información a través de múltiples plataformas"""
        print(f"\n🔍 Búsqueda cross-platform: '{query}'")
        print("=" * 70)
        
        results = []
        
        # Buscar en todos los servidores conectados
        for server in self.connected_servers:
            print(f"\n📡 Buscando en {server['name']}...")
            
            # Listar recursos
            resources = await self.list_resources(server['name'])
            
            # Leer cada recurso y buscar coincidencias (simplificado)
            for resource in resources[:2]:  # Limitar a 2 por servidor para demo
                content = await self.read_resource(resource['uri'])
                
                # Simulación de relevancia (en realidad usarías embeddings o búsqueda semántica)
                if any(word.lower() in content.lower() for word in query.split()):
                    results.append({
                        "platform": server['name'],
                        "resource": resource['name'],
                        "uri": resource['uri'],
                        "snippet": content[:100] + "...",
                        "relevance": 0.8  # Score simulado
                    })
        
        print(f"\n✅ Encontrados {len(results)} resultados relevantes")
        return results
    
    def synthesize_answer(self, query: str, results: List[Dict]) -> str:
        """Sintetizar respuesta basada en resultados de múltiples fuentes"""
        print(f"\n🤖 Sintetizando respuesta...")
        
        if not results:
            return "No encontré información relevante en las fuentes conectadas."
        
        # Agrupar por plataforma
        by_platform = {}
        for result in results:
            platform = result['platform']
            if platform not in by_platform:
                by_platform[platform] = []
            by_platform[platform].append(result)
        
        # Construir respuesta
        answer = f"Basándome en las fuentes conectadas, aquí está la información sobre '{query}':\n\n"
        
        for platform, items in by_platform.items():
            answer += f"📍 Desde {platform}:\n"
            for item in items:
                answer += f"   - {item['resource']}: {item['snippet']}\n"
            answer += "\n"
        
        return answer


async def main():
    """Demostración de cliente MCP multi-source"""
    print("=" * 70)
    print("Cliente MCP Multi-Source - Conexión Unificada a Múltiples Plataformas")
    print("=" * 70)
    
    # Crear cliente
    client = MCPClient()
    
    # Configuración de servidores MCP
    servers = [
        {
            "name": "slack-mcp",
            "type": "communication",
            "capabilities": ["list_channels", "read_messages", "search"]
        },
        {
            "name": "github-mcp",
            "type": "development",
            "capabilities": ["list_repos", "read_issues", "read_prs"]
        },
        {
            "name": "gdrive-mcp",
            "type": "storage",
            "capabilities": ["list_files", "read_files", "search"]
        }
    ]
    
    # Conectar a servidores
    for server_config in servers:
        await client.connect_to_server(server_config)
    
    print(f"\n✅ Cliente conectado a {len(client.connected_servers)} servidores MCP")
    
    # Usar el cliente para consultas cross-platform
    queries = [
        "información sobre el proyecto",
        "reportes del Q4",
    ]
    
    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"CONSULTA: {query}")
        print('=' * 70)
        
        # Buscar en todas las fuentes
        results = await client.query_cross_platform(query)
        
        # Sintetizar respuesta
        answer = client.synthesize_answer(query, results)
        
        print(f"\n💬 RESPUESTA:")
        print(answer)
    
    print("\n" + "=" * 70)
    print("✅ Demostración completada")
    print("=" * 70)
    
    print("""
💡 BENEFICIOS DE MCP:
   ✅ Una interfaz estándar para múltiples fuentes de datos
   ✅ No necesitas escribir integraciones específicas para cada plataforma
   ✅ Fácil agregar nuevas fuentes (solo conectar a nuevo servidor MCP)
   ✅ Búsqueda y síntesis cross-platform seamless

📝 NOTA: Este es un ejemplo educativo. En producción:
   - Usa el SDK oficial de MCP
   - Implementa autenticación real
   - Maneja errores y reconexiones
   - Usa embeddings para búsqueda semántica real
""")


if __name__ == "__main__":
    asyncio.run(main())
