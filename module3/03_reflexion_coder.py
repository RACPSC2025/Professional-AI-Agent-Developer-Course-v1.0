"""
🔴 NIVEL AVANZADO: CODIFICADOR AUTÓNOMO (REFLEXION)
---------------------------------------------------
Este script demuestra el patrón "Reflexion" (Self-Correction).
Es fundamental para tareas de codificación donde el primer intento suele fallar.

El ciclo es:
1. GENERAR: El LLM escribe código.
2. TESTEAR: Ejecutamos el código (aquí simulado con `exec` en un entorno controlado).
3. REFLEXIONAR: Si falla, el LLM lee el error y escribe una "Reflexión" de por qué falló.
4. RE-GENERAR: El LLM intenta de nuevo, usando su propia reflexión como guía.

Caso de Uso: Generación de código, escritura de emails complejos, tareas creativas.
"""

import sys
from io import StringIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- PROMPTS ---
gen_prompt = ChatPromptTemplate.from_template(
    """
    Eres un experto en Python.
    Tarea: Escribe una función en Python que resuelva lo siguiente: "{problem}"
    
    IMPORTANTE:
    - Solo devuelve el código. Nada de markdown ni explicaciones.
    - La función debe llamarse `solve()`.
    - Si tienes reflexiones previas de errores, ÚSALAS para no repetir el fallo.
    
    Reflexiones previas de errores:
    {reflections}
    """
)

reflect_prompt = ChatPromptTemplate.from_template(
    """
    Tu código anterior falló con este error:
    {error}
    
    Código fallido:
    {code}
    
    Analiza brevemente por qué falló y qué debes cambiar.
    Sé conciso. Ejemplo: "Olvidé importar math" o "Dividí por cero".
    """
)

# --- EJECUTOR SEGURO (SANDBOX SIMULADO) ---
def execute_python_code(code):
    """
    Ejecuta código Python dinámicamente y captura stdout/stderr.
    ⚠️ ADVERTENCIA: `exec` es peligroso en producción. Usar Docker/E2B en casos reales.
    """
    # Redirigir stdout para capturar prints
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Definimos un diccionario local para ejecutar
        local_scope = {}
        exec(code, {}, local_scope)
        
        # Verificar si existe la función solve
        if "solve" not in local_scope:
            return False, "Error: No definiste una función llamada `solve()`."
            
        # Ejecutar la función solve
        result = local_scope["solve"]()
        return True, result
        
    except Exception as e:
        return False, str(e)
    finally:
        sys.stdout = old_stdout

# --- BUCLE REFLEXION ---
def run_reflexion_coder(problem, max_retries=3):
    print(f"💻 PROBLEMA: {problem}\n")
    
    reflections = "Ninguna."
    
    for attempt in range(max_retries):
        print(f"--- INTENTO {attempt + 1} ---")
        
        # 1. Generar Código
        code = gen_prompt.invoke({"problem": problem, "reflections": reflections}).content
        # Limpieza básica de markdown si el modelo lo pone
        code = code.replace("```python", "").replace("```", "").strip()
        
        print(f"📝 Código Generado:\n{code}\n")
        
        # 2. Testear
        success, output = execute_python_code(code)
        
        if success:
            print(f"✅ ÉXITO! Resultado: {output}")
            return output
        else:
            print(f"❌ FALLO. Error: {output}")
            
            # 3. Reflexionar
            reflection = reflect_prompt.invoke({"error": output, "code": code}).content
            print(f"🤔 Reflexión: {reflection}\n")
            
            # Acumular reflexiones
            reflections += f"\n- Intento {attempt+1}: {reflection}"
            
    print("💀 Se acabaron los intentos.")
    return None

if __name__ == "__main__":
    # Problema trampa: Pedir algo que suele dar error si no se importan librerías
    problem = "Calcula la raíz cuadrada de 144 y multiplícala por PI. Imprime el resultado."
    run_reflexion_coder(problem)
