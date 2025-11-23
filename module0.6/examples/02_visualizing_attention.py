import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention_weights, tokens):
    """
    Genera un mapa de calor para visualizar los pesos de atención.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=True)
    plt.title("Self-Attention Weights Visualization")
    plt.xlabel("Key (Source)")
    plt.ylabel("Query (Target)")
    
    # Guardar la imagen en lugar de mostrarla (para entornos sin display)
    output_path = "attention_heatmap.png"
    plt.savefig(output_path)
    print(f"\n📊 Gráfico guardado como '{output_path}'")

def main():
    print("🎨 VISUALIZACIÓN DE ATENCIÓN CON MATPLOTLIB & SEABORN\n")
    
    tokens = ["The", "cat", "sat", "on", "mat"]
    
    # Simulamos una matriz de atención (5x5)
    # Imaginemos que "sat" tiene alta atención con "cat" y "mat"
    attention_weights = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0], # The -> The
        [0.1, 0.8, 0.1, 0.0, 0.0], # cat -> cat
        [0.0, 0.4, 0.4, 0.0, 0.2], # sat -> cat, sat, mat (RELACIÓN FUERTE)
        [0.0, 0.0, 0.1, 0.9, 0.0], # on -> on
        [0.1, 0.0, 0.1, 0.0, 0.8]  # mat -> mat
    ])
    
    print("Matriz de Atención Simulada:")
    print(attention_weights)
    
    print("\nGenerando mapa de calor...")
    try:
        plot_attention_heatmap(attention_weights, tokens)
        print("¡Éxito! Revisa el archivo generado para ver cómo el modelo conecta las palabras.")
    except Exception as e:
        print(f"Error al graficar: {e}")
        print("Asegúrate de tener matplotlib y seaborn instalados: pip install matplotlib seaborn")

if __name__ == "__main__":
    main()
