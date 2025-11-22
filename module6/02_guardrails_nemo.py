"""
02_guardrails_nemo.py
=====================
Este script demuestra cómo implementar NVIDIA NeMo Guardrails para proteger un agente.
NeMo Guardrails usa archivos de configuración (.co y .yml) para definir flujos de diálogo permitidos.

Conceptos clave:
1. Colang: Lenguaje para definir flujos de conversación.
2. Input Rails: Bloquear temas antes de llegar al LLM.
3. Output Rails: Filtrar respuestas inapropiadas.
4. Topical Rails: Mantener al bot en el tema (ej. solo hablar de tecnología).

Requisitos:
pip install nemoguardrails openai
"""

import os
import asyncio
from nemoguardrails import LLMRails, RailsConfig

# Configuración de Colang (definida inline para este ejemplo, normalmente va en un archivo .co)
COLANG_CONFIG = """
# Definir intenciones del usuario
define user ask about politics
  "¿Qué opinas de las elecciones?"
  "¿Quién debería ser presidente?"
  "¿Eres de izquierda o derecha?"

define user ask about competitors
  "¿Qué opinas de Claude?"
  "¿Es Gemini mejor que tú?"

define user ask technical question
  "¿Cómo funciona un Transformer?"
  "Explícame Python"

# Definir respuestas del bot
define bot refuse politics
  "Soy un asistente técnico enfocado en IA. No tengo opiniones políticas."

define bot refuse competitors
  "No puedo comentar sobre otros modelos. Estoy aquí para ayudarte con tu código."

# Definir flujos (Rails)
define flow politics
  user ask about politics
  bot refuse politics

define flow competitors
  user ask about competitors
  bot refuse competitors
"""

# Configuración YAML (definida inline)
YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

instructions:
  - type: general
    content: |
      Eres un asistente experto en Python y AI Agents.
      Debes ser útil, preciso y conciso.
"""

async def main():
    print("🛡️ Inicializando NeMo Guardrails...")
    
    # Crear configuración desde strings (en prod usarías from_path)
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG,
        yaml_content=YAML_CONFIG
    )
    
    # Inicializar Rails
    rails = LLMRails(config)
    
    # Casos de prueba
    test_inputs = [
        "¿Cómo funciona un Transformer?", # Safe
        "¿Qué opinas de las elecciones?", # Unsafe (Politics)
        "¿Es Claude mejor que tú?",       # Unsafe (Competitors)
    ]
    
    print("\n🧪 Ejecutando pruebas de Guardrails:\n")
    
    for user_input in test_inputs:
        print(f"👤 Usuario: {user_input}")
        
        # Generar respuesta protegida
        response = await rails.generate_async(messages=[{
            "role": "user",
            "content": user_input
        }])
        
        print(f"🤖 Agente:  {response['content']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
