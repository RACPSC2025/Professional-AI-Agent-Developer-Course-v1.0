"""
Ejemplo 2: Transformers Pipeline (Local Inference)
Nivel: 🟡 Intermedio
Objetivo: Descargar y ejecutar un modelo pequeño localmente usando `transformers`.

A diferencia de la API, aquí el modelo corre en TU máquina (CPU/GPU).
Usaremos 'GPT-2' (muy pequeño) para asegurar que corra en cualquier PC.

Requisitos:
- pip install transformers torch
"""

from transformers import pipeline, set_seed
import warnings

# Suprimir warnings de transformers para limpiar output
warnings.filterwarnings("ignore")

def run_local_pipeline():
    print("\n" + "="*60)
    print("💻 EJEMPLO 2: Transformers Pipeline (Local)")
    print("="*60 + "\n")

    # 1. Selección del Modelo
    # Usamos GPT-2 porque es muy ligero (~500MB) y corre rápido en CPU.
    # Para mejores resultados (si tienes GPU), prueba 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model_id = "gpt2" 
    
    print(f"📥 Descargando/Cargando modelo '{model_id}' en memoria local...")
    print("   (La primera vez esto tomará unos segundos/minutos)\n")

    # 2. Crear el Pipeline
    # 'text-generation' abstrae toda la complejidad (tokenización, modelo, decoding)
    generator = pipeline('text-generation', model=model_id)
    set_seed(42) # Para reproducibilidad

    # 3. Ejecutar Inferencia
    prompt = "The future of Artificial Intelligence is"
    
    print(f"📝 Prompt: '{prompt}'")
    print("⚙️  Generando...\n")

    outputs = generator(
        prompt, 
        max_length=50, 
        num_return_sequences=3,
        truncation=True
    )

    # 4. Mostrar Resultados
    print("✨ Resultados Generados:")
    for i, output in enumerate(outputs):
        print(f"\nOpción {i+1}:")
        print(f"'{output['generated_text']}'")

    print("\n" + "="*60)
    print("✅ Ventajas Local: Privacidad total, sin internet, costo cero.")
    print("❌ Desventajas Local: Consume RAM/CPU, descarga inicial.")

if __name__ == "__main__":
    run_local_pipeline()
