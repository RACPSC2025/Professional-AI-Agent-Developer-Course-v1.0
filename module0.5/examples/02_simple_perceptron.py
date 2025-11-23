import numpy as np

def sigmoid(x):
    """Función de activación Sigmoid: aplasta valores entre 0 y 1."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada de la Sigmoid para el cálculo del gradiente."""
    return x * (1 - x)

class SimplePerceptron:
    def __init__(self, input_size):
        # Inicializamos pesos aleatorios y sesgo (bias)
        np.random.seed(42) # Para reproducibilidad
        self.weights = 2 * np.random.random((input_size, 1)) - 1
        self.bias = 0

    def predict(self, inputs):
        # Forward Pass: Entradas * Pesos + Bias
        linear_output = np.dot(inputs, self.weights) + self.bias
        # Función de activación
        return sigmoid(linear_output)

    def train(self, training_inputs, training_outputs, epochs=10000):
        print(f"Entrenando por {epochs} épocas...")
        for epoch in range(epochs):
            # 1. Forward Pass
            output = self.predict(training_inputs)

            # 2. Cálculo del Error
            error = training_outputs - output

            # 3. Backpropagation (Ajuste de pesos)
            # Cuánto contribuyó cada peso al error
            adjustment = np.dot(training_inputs.T, error * sigmoid_derivative(output))
            
            # Actualizamos pesos
            self.weights += adjustment

def main():
    print("🧠 EL PERCEPTRÓN: LA NEURONA ARTIFICIAL\n")

    # Problema: Compuerta lógica XOR (No lineal) - El perceptrón simple falla aquí, 
    # pero usaremos una compuerta AND/OR simple para demostrar aprendizaje.
    # Vamos a enseñar la operación "AND": Solo es 1 si ambas entradas son 1.
    
    # Datos de entrenamiento (Inputs)
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Salidas esperadas (Outputs) para AND
    training_outputs = np.array([
        [0],
        [0],
        [0],
        [1]
    ]).T.T # Transponer para que sea columna (4,1)

    print("Inputs de entrenamiento:")
    print(training_inputs)
    print("Outputs esperados (AND):")
    print(training_outputs)

    # Crear Perceptrón
    perceptron = SimplePerceptron(input_size=2)
    
    print("\nPredicción antes de entrenar:")
    print(perceptron.predict(training_inputs))

    # Entrenar
    perceptron.train(training_inputs, training_outputs, epochs=20000)

    print("\nPredicción después de entrenar:")
    result = perceptron.predict(training_inputs)
    print(np.round(result, 3))
    
    print("\nInterpretación:")
    print(f"[0, 0] -> {result[0]} (Esperado: 0)")
    print(f"[0, 1] -> {result[1]} (Esperado: 0)")
    print(f"[1, 0] -> {result[2]} (Esperado: 0)")
    print(f"[1, 1] -> {result[3]} (Esperado: 1)")

if __name__ == "__main__":
    main()
