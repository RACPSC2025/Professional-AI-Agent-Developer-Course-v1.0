import time

def mock_cot_response(question):
    """
    Simula una respuesta usando Chain of Thought (CoT).
    En un modelo real, esto se logra con prompts como:
    "Let's think step by step" o "Explain your reasoning before answering".
    """
    print(f"❓ PREGUNTA: {question}")
    print("\n🧠 GENERANDO PENSAMIENTO (Chain of Thought)...\n")
    time.sleep(1) # Simular tiempo de "pensamiento"

    # Ejemplo de CoT para un problema matemático
    if "cafetería" in question.lower():
        thought_process = """
Thought:
1.  **Identificar el objetivo**: Calcular el total de manzanas que quedan.
2.  **Datos iniciales**:
    -   Inicio: 15 manzanas.
    -   Compradas: 5 manzanas.
    -   Vendidas: 2 manzanas.
    -   Comidas: 1 manzana.
3.  **Cálculos paso a paso**:
    -   Manzanas después de comprar: 15 + 5 = 20.
    -   Manzanas después de vender: 20 - 2 = 18.
    -   Manzanas después de comer: 18 - 1 = 17.
4.  **Verificación**: 15 + 5 - 2 - 1 = 17.
5.  **Formular respuesta final**.
        """
        final_answer = "Quedan 17 manzanas en la cafetería."
    
    # Ejemplo de CoT para razonamiento lógico
    elif "roger" in question.lower():
        thought_process = """
Thought:
1.  **Analizar la relación**: Roger es padre de John. John es padre de Peter.
2.  **Pregunta**: ¿Qué es Roger de Peter?
3.  **Inferencia lógica**:
    -   Si A es padre de B, y B es padre de C -> A es abuelo de C.
    -   A = Roger, B = John, C = Peter.
4.  **Conclusión**: Roger es el abuelo de Peter.
        """
        final_answer = "Roger es el abuelo de Peter."
    
    else:
        thought_process = "Thought: No tengo contexto suficiente para razonar sobre esto paso a paso."
        final_answer = "No puedo responder."

    print(thought_process)
    print(f"\n✅ RESPUESTA FINAL: {final_answer}")

def main():
    print("🔗 CHAIN OF THOUGHT (CoT) DEMO\n")
    print("CoT mejora drásticamente la capacidad de los LLMs para resolver problemas")
    print("de matemáticas, lógica y sentido común al forzar pasos intermedios.\n")

    # Caso 1: Matemáticas
    q1 = "Si la cafetería tiene 15 manzanas, compra 5 más, vende 2 y el dueño se come 1, ¿cuántas quedan?"
    mock_cot_response(q1)
    
    print("\n" + "-"*50 + "\n")

    # Caso 2: Lógica
    q2 = "Roger es el padre de John. John es el padre de Peter. ¿Qué relación tiene Roger con Peter?"
    mock_cot_response(q2)

if __name__ == "__main__":
    main()
