"""
Ejemplo 4: Agente Híbrido (Local + Cloud)
Nivel: 🔴 Avanzado
Objetivo: Crear un flujo que combine privacidad local con potencia en la nube.

Escenario:
1. Procesar datos sensibles (PII) localmente con Ollama (Privacidad).
2. Enviar datos sanitizados a un modelo potente en la nube (HuggingFace/OpenAI) para análisis creativo.

Requisitos:
- Ollama corriendo
- pip install ollama huggingface_hub
"""

import ollama
import os
from huggingface_hub import InferenceClient

def run_hybrid_agent():
    print("\n" + "="*60)
    print("🛡️☁️  EJEMPLO 4: Agente Híbrido (Local + Cloud)")
    print("="*60 + "\n")

    # --- CONFIGURACIÓN ---
    # Intentar encontrar un modelo local
    try:
        models = ollama.list()
        local_model = models['models'][0]['name'] # Usar el primero disponible
    except:
        print("❌ Ollama no detectado. Este demo requiere Ollama.")
        return

    # Cliente Cloud (HF)
    cloud_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2")

    # --- DATOS SENSIBLES SIMULADOS ---
    sensitive_email = """
    From: juan.perez@empresa.com
    To: soporte@banco.com
    Subject: Problema con tarjeta 4532-1234-5678-9012
    
    Hola, mi nombre es Juan Pérez, vivo en Calle Falsa 123, Madrid.
    Mi número de teléfono es +34 600 123 456.
    Tengo un cargo no reconocido de 500€ en mi tarjeta.
    """

    print("📨 Paso 1: Recibiendo datos sensibles (Email)")
    print(f"   (Contiene nombres, direcciones, tarjetas de crédito...)\n")

    # --- PASO 2: SANITIZACIÓN LOCAL (OLLAMA) ---
    print(f"🔒 Paso 2: Sanitizando localmente con {local_model}...")
    print("   (Los datos NO salen de tu PC en este paso)")

    sanitization_prompt = f"""
    You are a data privacy agent. Replace all PII (Personally Identifiable Information) 
    like names, addresses, phone numbers, and credit card numbers with placeholders 
    like [NAME], [ADDRESS], [PHONE], [CARD].
    
    Do NOT summarize. Just output the sanitized text.
    
    Text to sanitize:
    {sensitive_email}
    """

    local_response = ollama.chat(model=local_model, messages=[
        {'role': 'user', 'content': sanitization_prompt}
    ])
    
    sanitized_text = local_response['message']['content']
    print(f"\n✅ Texto Sanitizado:\n{sanitized_text}\n")

    # --- PASO 3: ANÁLISIS EN LA NUBE (HUGGINGFACE) ---
    print("☁️  Paso 3: Enviando texto seguro a la nube para análisis de sentimiento...")
    
    analysis_prompt = f"""
    [INST] Analyze the sentiment and urgency of the following customer email. 
    Suggest a polite response strategy.
    
    Email:
    {sanitized_text}
    [/INST]
    """

    try:
        cloud_response = cloud_client.text_generation(
            analysis_prompt,
            max_new_tokens=200
        )
        print(f"\n✨ Estrategia sugerida (Cloud AI):\n{cloud_response}")
        
    except Exception as e:
        print(f"❌ Error en cloud: {e}")

    print("\n" + "="*60)
    print("🏆 Conclusión: Logramos inteligencia avanzada sin exponer datos privados.")

if __name__ == "__main__":
    run_hybrid_agent()
