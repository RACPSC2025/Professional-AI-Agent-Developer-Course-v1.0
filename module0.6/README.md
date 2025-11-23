# Módulo 0.6: Applied Data Science for AI

![Module 0.6 Banner](../images/module0.6_banner.png)

> "Data is the new oil. It's valuable, but if unrefined it cannot really be used." — Clive Humby

## 📌 Introducción

Antes de construir agentes inteligentes, necesitas dominar la materia prima de la IA: **los datos**. En este módulo, profundizaremos en el stack científico de Python (**NumPy, Pandas, Matplotlib**) no solo como herramientas generales, sino enfocadas específicamente en las necesidades de la Inteligencia Artificial: manipulación de tensores, limpieza de datasets para RAG y visualización de métricas de modelos.

---

## 📚 Índice

1. [NumPy: El Motor Numérico](#1-numpy-el-motor-numérico)
2. [Pandas: Manipulación de Datos](#2-pandas-manipulación-de-datos)
3. [Matplotlib & Seaborn: Visualización](#3-matplotlib--seaborn-visualización)
4. [Recursos y Datasets Gratuitos](#4-recursos-y-datasets-gratuitos)

---

## 1. NumPy: El Motor Numérico

NumPy (Numerical Python) es la base sobre la que se construye todo el ecosistema de Deep Learning (PyTorch, TensorFlow). Entender NumPy es entender cómo "piensan" las máquinas: en vectores y matrices.

### 🧠 The Basics of NumPy Arrays

A diferencia de las listas de Python, los arrays de NumPy son homogéneos y contiguos en memoria, lo que permite operaciones vectorizadas ultrarrápidas.

```python
import numpy as np

# Crear un array (vector)
vector = np.array([1, 2, 3], dtype='float32')

# Crear una matriz (tensor 2D)
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

print(f"Shape: {matrix.shape}")  # (2, 3)
print(f"Dimensiones: {matrix.ndim}") # 2
print(f"Tipo de dato: {matrix.dtype}") # float32 (Estándar en IA)

### 🆕 NumPy 2.0 (2025 Standard): StringDType
Para NLP, NumPy 2.0 introdujo `StringDType`, mucho más eficiente para texto variable que el antiguo `U` (Unicode fijo).

```python
# NumPy 2.0+
text_data = np.array(["chat", "user", "assistant"], dtype=np.StringDType())
```
```

### ⚡ Computation: Universal Functions (UFuncs)

Olvídate de los bucles `for`. En NumPy, las operaciones se aplican a todo el array simultáneamente.

```python
x = np.arange(1000000)

# Lento (Python puro)
# [i * 2 for i in x] 

# Rápido (NumPy Vectorizado)
x * 2  # Se aplica a todos los elementos a la vez
```

### 📡 Broadcasting: La Magia de las Dimensiones

Broadcasting permite operar arrays de diferentes formas. Es crucial para entender cómo se añaden los *bias* en una red neuronal.

```python
A = np.array([[1, 2, 3], 
              [4, 5, 6]]) # Shape (2, 3)
b = np.array([10, 20, 30]) # Shape (3,)

# b se "estira" virtualmente para sumar a cada fila de A
result = A + b 
# [[11, 22, 33],
#  [14, 25, 36]]
```

### 🔍 Indexing & Slicing Avanzado

```python
data = np.random.rand(5, 5)

# Fancy Indexing: Seleccionar filas específicas
indices = [0, 2, 4]
selected_rows = data[indices]

# Boolean Masking: Filtrar datos
mask = data > 0.5
filtered_data = data[mask] # Solo valores mayores a 0.5
```

### 📐 Álgebra Lineal (Producto Punto)

La operación más importante en IA (Attention mechanism, Dense layers).

```python
# Producto punto (Dot Product)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

dot_product = a @ b  # O np.dot(a, b)
```

---

## 2. Pandas: Manipulación de Datos

Pandas es esencial para preparar los datos que tus agentes consumirán (RAG) o para analizar su comportamiento.

### 🐼 Pandas Objects: Series & DataFrames

```python
import pandas as pd

# DataFrame: Tabla de datos
df = pd.DataFrame({
    'prompt': ['Explain AI', 'Write code', 'Translate'],
    'tokens': [150, 300, 50],
    'model': ['gpt-4', 'claude-3', 'gpt-4']
})
```

### 🚀 Pandas 3.0+ & PyArrow Backend (2025)

En 2025, el estándar para IA es usar el backend **PyArrow**. Es 10x más rápido y usa 70% menos memoria para texto (crucial para datasets de LLMs).

```python
# Activar Copy-on-Write (Default en Pandas 3.0)
pd.options.mode.copy_on_write = True

# Cargar dataset con motor PyArrow (Mucho más rápido)
df = pd.read_csv("large_dataset.csv", engine="pyarrow", dtype_backend="pyarrow")

# Los strings ahora son 'string[pyarrow]', no 'object'
print(df.dtypes) 
```

### 🧹 Data Cleaning & Handling Missing Data

Los datos reales son sucios. Antes de meterlos a un vector database, límpialos.

```python
# Detectar nulos
print(df.isnull().sum())

# Rellenar nulos (Imputation)
df['tokens'] = df['tokens'].fillna(0)

# Eliminar filas corruptas
df_clean = df.dropna()
```

### 🎯 Selection & Filtering

```python
# Seleccionar columnas
prompts = df['prompt']

# Filtrar filas (Querying)
expensive_calls = df[df['tokens'] > 200]

# Filtrado complejo
gpt4_calls = df[(df['model'] == 'gpt-4') & (df['tokens'] > 100)]
```

### 📊 Aggregation & Grouping

Analiza el rendimiento de tus agentes.

```python
# Costo promedio por modelo
avg_tokens = df.groupby('model')['tokens'].mean()
print(avg_tokens)

# Pivot Tables (Resumen multidimensional)
pivot = df.pivot_table(values='tokens', index='model', aggfunc=['mean', 'max'])
```

### 🔗 Combining Datasets

Unir datos de diferentes fuentes (ej. logs de chat + feedback de usuarios).

```python
# Merge (como SQL JOIN)
merged_df = pd.merge(logs_df, feedback_df, on='request_id', how='left')
```

---

## 3. Matplotlib & Seaborn: Visualización

"Una imagen vale más que mil tokens". Visualiza el comportamiento de tus modelos.

### 📈 Simple Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración profesional
sns.set_theme(style="whitegrid")

# Line Plot (Curvas de entrenamiento/Loss)
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, label='Training Loss', color='blue')
plt.plot(epochs, val_loss_values, label='Validation Loss', color='red', linestyle='--')
plt.title("Curva de Aprendizaje del Modelo")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

### 🌌 Scatter Plots (Embeddings)

Visualiza clusters de documentos en tu base de datos vectorial (usando t-SNE o PCA).

```python
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title("Visualización de Embeddings (Espacio Semántico)")
plt.colorbar()
plt.show()
```

### 📊 Histograms (Distribución de Datos)

Entiende la longitud de tus contextos.

```python
sns.histplot(df['token_count'], bins=30, kde=True, color='purple')
plt.title("Distribución de Longitud de Prompts")
plt.xlabel("Tokens")
plt.show()
```

### 🔥 Heatmaps (Matrices de Atención/Confusión)

Visualiza qué partes del input está "mirando" el modelo.

```python
attention_matrix = np.random.rand(10, 10) # Ejemplo
sns.heatmap(attention_matrix, cmap='Reds')
plt.title("Mapa de Atención")
plt.show()
```

---

## 4. Recursos y Datasets Gratuitos

Para practicar Data Science y entrenar/evaluar tus agentes, necesitas datos. Aquí tienes una colección de fuentes gratuitas de alta calidad.

### 📂 Repositorios de CSVs y Datasets

1.  **Math Dept. CSV Collection**: Datasets clásicos limpios para pruebas rápidas.
    - [🔗 Link](https://people.math.sc.edu/Burkardt/datasets/csv/csv.html)

2.  **GitHub CSV Collection**: Recopilación de CSVs para Data Science y ML.
    - [🔗 Link](https://github.com/sachin365123/CSV-files-for-Data-Science-and-Machine-Learning)

3.  **DataQuest Free Datasets**: Lista curada de datasets interesantes para proyectos.
    - [🔗 Link](https://www.dataquest.io/blog/free-datasets-for-projects/)

4.  **OpenDataBay AI/ML**: Datasets específicos para entrenamiento de modelos.
    - [🔗 Link](https://www.opendatabay.com/data/ai-ml/19c7e7a0-70b8-46fc-94e8-2ec8536a1c47)

5.  **HuggingFace Datasets**: El estándar de oro para NLP y LLMs.
    - [🔗 Link](https://huggingface.co/datasets)

6.  **Kaggle Datasets**: La comunidad más grande de Data Science.
    - [🔗 Link](https://www.kaggle.com/datasets)

---

## 🚀 Siguiente Paso

Ahora que dominas la manipulación y visualización de datos, estás listo para entender los modelos que procesan estos datos.

➡️ **[Ir al Módulo 1: LLMs y Mentalidad Agéntica](../module1/README.md)**

<div align="center">

**[⬅️ Módulo 0.5: Fundamentos Matemáticos](../module0.5/README.md)** | **[🏠 Inicio](../README.md)**

</div>
