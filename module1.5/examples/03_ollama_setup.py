"""
Ejemplo 3: Ollama Setup & Python Interaction
Nivel: 🟡 Intermedio
Objetivo: Interactuar con un modelo corriendo en Ollama desde Python.

Ollama debe estar instalado y corriendo en tu máquina.
Descarga: https://ollama.com/

Requisitos:
- Ollama instalado y corriendo (`ollama serve`)
- Modelo descargado (`ollama pull llama3` o `ollama pull tinyllama`)
- pip install ollama
"""

import ollama
import sys

def run_ollama_demo():
    print("\n" + "="*60)
    print("🦙 EJEMPLO 3: Ollama Python Client")
    print("="*60 + "\n")

    # 1. Verificar conexión con Ollama
    try:
        print("📡 Conectando con Ollama local...")
        models_info = ollama.list()
        print("✅ Conexión exitosa.")
        
        # Listar modelos disponibles
        available_models = [m['name'] for m in models_info['models']]
        print(f"📂 Modelos instalados: {available_models}")
        
        if not available_models:
            print("❌ No hay modelos instalados. Ejecuta 'ollama pull llama3' en tu terminal.")
            return

    except Exception as e:
        print(f"❌ Error conectando a Ollama: {e}")
        print("¿Está corriendo Ollama? Ejecuta 'ollama serve' o abre la app.")
        return

    # 2. Seleccionar modelo (usar el primero disponible o uno específico)
    # Preferimos llama3, mistral, o tinyllama
    preferred_models = ['llama3', 'mistral', 'tinyllama', 'gemma:2b']
    selected_model = None
    
    # Buscar coincidencia parcial
    for pref in preferred_models:
        for avail in available_models:
            if pref in avail:
                selected_model = avail
                break
        if selected_model: break
    
    if not selected_model:
        selected_model = available_models[0] # Fallback al primero

    print(f"\n🤖 Usando modelo: {selected_model}")

    # 3. Chat Simple (No streaming)
    prompt = "Explain quantum computing in one sentence."
    print(f"\n📝 Prompt: {prompt}")
    print("⏳ Pensando...")
    
    response = ollama.chat(model=selected_model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    
    print(f"✨ Respuesta:\n{response['message']['content']}")

    # 4. Chat con Streaming (Como ChatGPT)
    print("\n" + "-"*40 + "\n")
    print("🌊 Probando Streaming (Escribe una historia corta)...")
    
    stream_prompt = "Write a very short story about a robot who loves gardening."
    
    stream = ollama.chat(
        model=selected_model,
        messages=[{'role': 'user', 'content': stream_prompt}],
        stream=True,
    )

    print("✨ Respuesta (Streaming):")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")

if __name__ == "__main__":
    run_ollama_demo()
