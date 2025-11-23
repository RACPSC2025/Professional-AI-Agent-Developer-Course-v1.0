import numpy as np

def softmax(x):
    """Calcula la función Softmax para un array numpy."""
    e_x = np.exp(x - np.max(x)) # Restamos max para estabilidad numérica
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(query, key, value):
    """
    Implementación manual de Scaled Dot-Product Attention.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = query.shape[-1]
    
    # 1. Producto Punto (Matmul) entre Query y Key Transpuesta
    scores = np.matmul(query, key.T)
    
    # 2. Escalamiento (Scaling)
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. Softmax para obtener probabilidades (pesos de atención)
    attention_weights = softmax(scaled_scores)
    
    # 4. Multiplicación por Value
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

def main():
    print("🧮 SIMULACIÓN DE SELF-ATTENTION CON NUMPY\n")
    
    # Simulamos embeddings para 3 palabras: "AI", "Agents", "Rules"
    # Dimensión del embedding = 4
    # En la realidad, estos números se aprenden durante el entrenamiento.
    inputs = np.array([
        [1.0, 0.0, 1.0, 0.0], # AI
        [0.0, 1.0, 0.0, 1.0], # Agents
        [1.0, 1.0, 1.0, 1.0]  # Rules
    ])
    
    print("Input Embeddings (3 palabras, dim=4):")
    print(inputs)
    
    # En self-attention simple, Q, K, V suelen ser proyecciones del input.
    # Aquí usaremos el input directo para simplificar.
    Q = inputs
    K = inputs
    V = inputs
    
    print("\nCalculando Atención...")
    output, weights = self_attention(Q, K, V)
    
    print("\n✅ Pesos de Atención (Attention Weights):")
    print("Muestra cuánto se enfoca cada palabra en las otras.")
    print(np.round(weights, 2))
    
    print("\n✅ Salida (Contextualized Embeddings):")
    print("Nuevas representaciones que mezclan información basada en la atención.")
    print(np.round(output, 2))

if __name__ == "__main__":
    main()
