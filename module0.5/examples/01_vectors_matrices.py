import numpy as np

def main():
    print("🧮 FUNDAMENTOS DE ÁLGEBRA LINEAL CON NUMPY\n")

    # 1. VECTORES
    # Representan puntos en el espacio o características de un objeto.
    # En IA, las palabras se convierten en vectores (Embeddings).
    v1 = np.array([2, 3])
    v2 = np.array([4, 1])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")

    # Suma de vectores (Desplazamiento en el espacio)
    v_sum = v1 + v2
    print(f"\nSuma (v1 + v2): {v_sum}")

    # 2. PRODUCTO PUNTO (DOT PRODUCT)
    # Mide la similitud entre dos vectores.
    # Si es 0, son ortogonales (no tienen nada que ver).
    # Si es alto, apuntan en la misma dirección (son similares).
    dot_product = np.dot(v1, v2)
    print(f"\nProducto Punto (v1 . v2): {dot_product}")
    print("Cálculo manual: (2*4) + (3*1) = 8 + 3 = 11")

    # 3. MATRICES
    # Transformaciones lineales. Una red neuronal es una cadena de multiplicaciones de matrices.
    M = np.array([
        [1, 2],
        [3, 4]
    ])
    print(f"\nMatriz M:\n{M}")

    # Multiplicación Matriz-Vector
    # Transformamos el vector v1 usando la matriz M.
    transformed_v1 = np.dot(M, v1)
    print(f"\nTransformación de v1 por M (M . v1): {transformed_v1}")
    print("Cálculo: [1*2 + 2*3, 3*2 + 4*3] = [2+6, 6+12] = [8, 18]")

    # 4. EJEMPLO REAL: SIMILITUD DE TEXTO (SIMPLIFICADO)
    print("\n--- EJEMPLO: SIMILITUD SEMÁNTICA ---")
    # Imaginemos embeddings de 3 dimensiones para palabras
    king = np.array([0.9, 0.1, 0.5])
    man = np.array([0.8, 0.1, 0.4])
    apple = np.array([0.1, 0.9, 0.2])

    print(f"King: {king}")
    print(f"Man: {man}")
    print(f"Apple: {apple}")

    # Similitud Coseno (Producto punto normalizado)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_king_man = cosine_similarity(king, man)
    sim_king_apple = cosine_similarity(king, apple)

    print(f"\nSimilitud King vs Man: {sim_king_man:.4f} (Alta)")
    print(f"Similitud King vs Apple: {sim_king_apple:.4f} (Baja)")

if __name__ == "__main__":
    main()
