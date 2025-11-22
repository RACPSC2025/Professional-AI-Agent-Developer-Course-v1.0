# Módulo 3: Arquitecturas Cognitivas y Patrones de Diseño

## 🎯 Objetivos del Módulo
Un LLM por sí solo es solo un predictor de texto. Una **Arquitectura Cognitiva** es la estructura que le permite "razonar". En este módulo, aprenderás los patrones de diseño que convierten a un modelo tonto en un agente inteligente capaz de corregirse a sí mismo.

## 📚 Conceptos Clave

### 1. ReAct (Reason + Act)
-   El patrón fundacional.
-   Ciclo: **Pensamiento** ("Debo buscar X") -> **Acción** (Ejecutar búsqueda) -> **Observación** (Ver resultados) -> **Repetir**.

### 2. Chain of Thought (CoT) & Tree of Thoughts (ToT)
-   **CoT:** Forzar al modelo a explicar su razonamiento paso a paso ("Pensemos paso a paso...").
-   **ToT:** Explorar múltiples ramas de razonamiento y descartar las que no llevan a solución.

### 3. Reflexion (Self-Correction)
-   La capacidad crítica. El agente genera una salida, la evalúa él mismo ("¿Esto cumple con lo que pidió el usuario?"), y si no, la corrige.
-   Esencial para generación de código y tareas creativas.

### 4. Plan-and-Solve
-   Para tareas complejas, primero generar un plan explícito ("Paso 1, Paso 2...") y luego ejecutarlo. Evita que el agente se pierda en los detalles.

## 🛠️ Proyectos Prácticos (Niveles de Dificultad)

### 🟢 Nivel Básico: Verificador de Hechos (ReAct)
-   **Concepto:** Patrón Reason + Act.
-   **Misión:** Validar una afirmación compleja ("¿Es verdad que el inventor del transistor ganó dos premios Nobel?") descomponiéndola en pasos de búsqueda secuenciales.

### 🟡 Nivel Intermedio: Planificador de Viajes (Plan-and-Solve)
-   **Concepto:** Separar Planificación de Ejecución.
-   **Misión:** Generar primero un itinerario de alto nivel ("Día 1: Tokyo, Día 2: Kyoto") y LUEGO llamar a herramientas para llenar los detalles de cada día (Hoteles, Trenes).

### 🔴 Nivel Avanzado: Codificador Autónomo (Reflexion)
-   **Concepto:** Self-Correction Loop.
-   **Misión:**
    1.  Escribe código Python para resolver un problema.
    2.  Ejecuta el código en un entorno seguro.
    3.  Si hay error, lee el Traceback.
    4.  **Reflexiona:** "¿Por qué falló?".
    5.  Reescribe el código y reintenta hasta el éxito.

## 💻 Snippet: ReAct Loop (Conceptual)

```python
# Bucle ReAct simplificado
pregunta = "¿Quién es el CEO de Microsoft y qué edad tiene?"
historial = [pregunta]

while not respuesta_final:
    pensamiento = llm.generar_pensamiento(historial)
    if "ACCIÓN:" in pensamiento:
        herramienta, input = parsear(pensamiento)
        observacion = ejecutar_herramienta(herramienta, input)
        historial.append(f"OBSERVACIÓN: {observacion}")
    else:
        respuesta_final = pensamiento

print(respuesta_final)
```
