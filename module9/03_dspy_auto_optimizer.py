"""
03_dspy_auto_optimizer.py
=========================
Este script introduce DSPy, un framework que "programa" los prompts por ti.
En lugar de escribir prompts manuales, definimos una "Firma" y usamos un "Optimizador"
para que DSPy encuentre el mejor prompt y ejemplos automáticamente.

Caso: Clasificación de Sentimientos (Simple para demostración).

Requisitos:
pip install dspy-ai
"""

import dspy
from dspy.teleprompt import BootstrapFewShot

# Configuración (Usamos OpenAI)
# dspy.settings.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo'))

# Mock para que el script corra sin API Key real si es necesario, 
# pero en prod descomentar la línea de arriba.
class MockLM(dspy.LM):
    def __init__(self):
        super().__init__("mock")
        self.history = []
    def basic_request(self, prompt, **kwargs):
        return ["Positive"] # Simula respuesta siempre positiva

lm = MockLM()
dspy.settings.configure(lm=lm)


# 1. Definir la Firma (Signature)
# Esto define QUÉ queremos (Input -> Output), no CÓMO pedirlo.
class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a sentence as Positive, Negative, or Neutral."""
    sentence = dspy.InputField()
    sentiment = dspy.OutputField()

# 2. Definir el Módulo
# ChainOfThought añade automáticamente "Let's think step by step"
class SentimentModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(SentimentClassifier)
    
    def forward(self, sentence):
        return self.prog(sentence=sentence)

# 3. Datos de Entrenamiento (Pequeño set)
trainset = [
    dspy.Example(sentence="I love this product", sentiment="Positive").with_inputs("sentence"),
    dspy.Example(sentence="This is terrible", sentiment="Negative").with_inputs("sentence"),
    dspy.Example(sentence="It's okay, nothing special", sentiment="Neutral").with_inputs("sentence"),
]

# 4. Optimización (La Magia ✨)
def main():
    print("🧬 Iniciando DSPy Optimizer...")
    
    # Definir métrica de éxito (Exact Match)
    def validate_answer(example, pred, trace=None):
        return example.sentiment == pred.sentiment

    # El Teleprompter "aprende" de los datos y optimiza el prompt
    teleprompter = BootstrapFewShot(metric=validate_answer)
    
    print("   Optimizando prompts y seleccionando few-shot examples...")
    # En un entorno real con LLM conectado, esto probaría variaciones
    compiled_program = teleprompter.compile(SentimentModule(), trainset=trainset)
    
    # 5. Probar
    test_sentence = "I am extremely happy with the results!"
    print(f"\n🧪 Probando con: '{test_sentence}'")
    
    pred = compiled_program(test_sentence)
    
    print(f"🤖 Predicción: {pred.sentiment}")
    print("\n📝 Prompt Optimizado (Simulado):")
    # DSPy habría construido un prompt con instrucciones + ejemplos seleccionados
    print("(DSPy ha inyectado automáticamente ejemplos relevantes del trainset en el prompt)")

if __name__ == "__main__":
    main()
