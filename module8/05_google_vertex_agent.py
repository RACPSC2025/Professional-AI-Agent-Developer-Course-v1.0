"""
05_google_vertex_agent.py
=========================
Este script ilustra la estructura de un agente usando el ecosistema de Google Vertex AI.
Se enfoca en el uso de modelos Gemini y la definición de Herramientas (Tools).
Google Vertex AI Agents está diseñado para escalar masivamente en la nube.

Caso de Uso: Agente de Planificación de Viajes.

Requisitos:
pip install google-cloud-aiplatform
"""

# Nota: Este código es conceptual y requiere credenciales de GCP reales para ejecutarse.
# Se presenta como una plantilla educativa de "Best Practices".

class VertexAgent:
    def __init__(self, model_name="gemini-1.5-pro"):
        self.model_name = model_name
        self.tools = []
        print(f"☁️ Inicializando Vertex AI Agent con {model_name}...")

    def add_tool(self, func):
        """Registra una función Python como herramienta para el modelo."""
        self.tools.append(func)
        print(f"   🛠️ Herramienta agregada: {func.__name__}")

    def chat(self, user_query):
        """Simula el bucle de razonamiento (ReAct) del modelo."""
        print(f"\n👤 Usuario: {user_query}")
        print("🤖 Agente: Pensando...")
        
        # Simulación de lógica interna
        if "vuelos" in user_query.lower():
            print("   > Decisión: Necesito buscar vuelos.")
            result = self.search_flights("Madrid", "Tokyo")
            return f"He encontrado vuelos a Tokyo desde Madrid por 800€. {result}"
        
        return "Soy un agente de viajes. ¿En qué puedo ayudarte?"

    # Herramientas
    def search_flights(self, origin, dest):
        return "Vuelo IB6800 disponible."

def main():
    # 1. Crear el Agente
    agent = VertexAgent()

    # 2. Definir Herramientas
    def search_hotels(location: str):
        """Busca hoteles en una ubicación."""
        pass

    agent.add_tool(search_hotels)
    agent.add_tool(agent.search_flights)

    # 3. Interactuar
    response = agent.chat("Quiero buscar vuelos de Madrid a Tokyo para mañana.")
    print(f"🤖 Respuesta Final: {response}")

if __name__ == "__main__":
    print("Nota: Para ejecutar esto realmente, necesitas configurar 'gcloud auth application-default login'")
    main()
