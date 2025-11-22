"""
02_autogen_coding_team.py
=========================
Este script demuestra cómo usar Microsoft AutoGen para crear un equipo de desarrollo de software.
AutoGen permite que los agentes conversen entre sí y EJECUTEN código real.

Caso de Uso: Resolver un problema matemático escribiendo y ejecutando Python.

Arquitectura:
- UserProxy: Actúa como el "Jefe" y ejecutor de código. Da la tarea y ejecuta lo que el Coder escribe.
- Assistant (Coder): Escribe el código Python para resolver la tarea.

Requisitos:
pip install pyautogen
"""

import os
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Configuración (Simulada para el ejemplo, requiere API Key real)
# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ.get("OPENAI_API_KEY")}],
    "seed": 42,
    "temperature": 0
}

def main():
    print("🤖 Iniciando AutoGen Coding Team...\n")

    # 1. Crear el Agente Asistente (El Coder)
    # Este agente recibe la tarea y escribe código para solucionarla.
    assistant = AssistantAgent(
        name="Coder_Agent",
        llm_config=llm_config,
        system_message="""Eres un experto en Python.
        Escribe código para resolver las tareas del usuario.
        Imprime los resultados en stdout.
        Si el código falla, analízalo y propón una corrección.
        Cuando la tarea esté resuelta, responde con TERMINATE."""
    )

    # 2. Crear el Agente Proxy de Usuario (El Ejecutor)
    # Este agente ejecuta el código que escribe el Assistant y le devuelve el output.
    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER", # Automatizado completamente
        max_consecutive_auto_reply=5, # Limite de iteraciones para evitar bucles infinitos
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding_output", # Carpeta donde se guardan los scripts
            "use_docker": False # Ejecutar localmente (CUIDADO en prod)
        }
    )

    # 3. Iniciar la conversación
    task = """
    Escribe un script en Python que:
    1. Calcule la secuencia de Fibonacci hasta el número 50.
    2. Guarde los números en un archivo 'fibonacci.txt'.
    3. Imprima los últimos 5 números de la secuencia.
    """
    
    print(f"Tarea: {task}")
    
    user_proxy.initiate_chat(
        assistant,
        message=task
    )

if __name__ == "__main__":
    # Nota: Para correr esto necesitas una API Key válida de OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        main()
    else:
        print("⚠️ Por favor configura la variable de entorno OPENAI_API_KEY para ejecutar este ejemplo.")
