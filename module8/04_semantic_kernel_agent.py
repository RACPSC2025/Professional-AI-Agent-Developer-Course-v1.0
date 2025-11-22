"""
04_semantic_kernel_agent.py
===========================
Este script demuestra cómo usar Microsoft Semantic Kernel para crear un agente
que puede planificar y ejecutar tareas usando "Plugins".
Semantic Kernel es ideal para aplicaciones empresariales que requieren integración con .NET/Python.

Caso de Uso: Asistente de Productividad que gestiona tiempo y correos.

Requisitos:
pip install semantic-kernel
"""

import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

# 1. Definir Plugins (Habilidades del Agente)

class TimePlugin:
    """Plugin para manejar el tiempo y agenda."""
    
    @kernel_function(description="Obtiene la fecha y hora actual.")
    def get_current_time(self) -> str:
        return "2025-10-25 14:30:00" # Simulado

    @kernel_function(description="Agenda una reunión.")
    def schedule_meeting(self, title: str, time: str) -> str:
        return f"✅ Reunión '{title}' agendada para {time}."

class EmailPlugin:
    """Plugin para enviar correos."""
    
    @kernel_function(description="Envía un correo electrónico.")
    def send_email(self, recipient: str, subject: str, body: str) -> str:
        return f"📧 Correo enviado a {recipient} con asunto '{subject}'."

# 2. Configurar el Kernel

async def main():
    print("🏢 Iniciando Microsoft Semantic Kernel Agent...\n")
    
    kernel = Kernel()

    # Configurar servicio de IA (Simulado o Real)
    # En prod: kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4", api_key="..."))
    
    # Registrar Plugins
    kernel.add_plugin(TimePlugin(), plugin_name="Time")
    kernel.add_plugin(EmailPlugin(), plugin_name="Email")

    # 3. Ejecución Directa (Function Calling)
    # En SK, los agentes suelen usar "Planners" para decidir qué función llamar.
    # Aquí simulamos la invocación directa para claridad.
    
    print("--- Paso 1: Consultar Agenda ---")
    # El LLM decidiría llamar a esta función:
    time_plugin = kernel.get_plugin("Time")
    current_time = await time_plugin["get_current_time"].invoke(kernel)
    print(f"Hora actual: {current_time}")

    print("\n--- Paso 2: Agendar Reunión ---")
    meeting_result = await time_plugin["schedule_meeting"].invoke(
        kernel, title="Revisión de Proyecto", time="15:00"
    )
    print(meeting_result)

    print("\n--- Paso 3: Notificar por Correo ---")
    email_plugin = kernel.get_plugin("Email")
    email_result = await email_plugin["send_email"].invoke(
        kernel, 
        recipient="boss@company.com", 
        subject="Reunión Agendada", 
        body="He agendado la revisión para las 15:00."
    )
    print(email_result)
    
    print("\n✨ Flujo Empresarial Completado.")

if __name__ == "__main__":
    asyncio.run(main())
