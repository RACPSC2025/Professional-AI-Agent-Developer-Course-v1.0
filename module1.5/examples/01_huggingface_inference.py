"""
Ejemplo 1: HuggingFace Inference API (Serverless)
Nivel: 🟢 Básico
Objetivo: Usar modelos Open Source sin descargarlos localmente.

La Inference API de HuggingFace permite enviar inputs a modelos alojados
en sus servidores. Es ideal para prototipos rápidos y gratuitos.

Requisitos:
- pip install huggingface_hub
- HUGGINGFACE_API_TOKEN (opcional para rate limits más altos)
"""

import os
from huggingface_hub import InferenceClient

def run_inference_demo():
    print("\n" + "="*60)
    print("🤗 EJEMPLO 1: HuggingFace Inference API")
    print("="*60 + "\n")

    # 1. Configuración
    # Puedes obtener tu token en: https://huggingface.co/settings/tokens
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    if not token:
        print("⚠️  Nota: Ejecutando sin token (rate limits bajos).")
        print("   Para producción, configura HUGGINGFACE_API_TOKEN environment variable.\n")

    # 2. Cliente de Inferencia
    # Usaremos un modelo popular y ligero: Mistral-7B-Instruct
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    client = InferenceClient(model=model_id, token=token)

    # 3. Tarea: Generación de Texto
    prompt = """
    <s>[INST] Explica qué es un "Agente de IA" en una frase simple y divertida para un niño de 10 años. [/INST]
    """
    
    print(f"🤖 Modelo: {model_id}")
    print(f"📝 Prompt: {prompt.strip()}\n")
    print("⏳ Generando respuesta en la nube...\n")

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        print(f"✨ Respuesta:\n{response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Posible causa: Modelo cargando (cold start) o rate limit excedido.")

    # 4. Tarea: Summarization (usando otro modelo especializado)
    print("\n" + "-"*40 + "\n")
    print("📚 Probando tarea de Resumen (Summarization)...")
    
    summ_model = "facebook/bart-large-cnn"
    summ_client = InferenceClient(model=summ_model, token=token)
    
    text_to_summarize = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
    """
    
    try:
        summary = summ_client.summarization(text_to_summarize)
        print(f"📝 Texto original: {len(text_to_summarize)} caracteres")
        print(f"✨ Resumen: {summary.summary_text}")
        
    except Exception as e:
        print(f"❌ Error en resumen: {e}")

if __name__ == "__main__":
    run_inference_demo()
