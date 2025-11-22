"""
🟡 NIVEL INTERMEDIO: PLANIFICADOR DE VIAJES (PLAN-AND-SOLVE)
------------------------------------------------------------
Este script demuestra el patrón "Plan-and-Solve" (o Plan-and-Execute).
A diferencia de ReAct (que piensa paso a paso), aquí:
1. PLANNER: Genera un plan completo primero.
2. EXECUTOR: Ejecuta cada paso del plan secuencialmente.

Ventaja: Evita que el agente se "pierda" en detalles y olvide el objetivo principal.
Caso de Uso: Tareas largas como planear un viaje, escribir un libro, o proyectos de código.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 1. EL PLANIFICADOR (PLANNER) ---
planner_prompt = ChatPromptTemplate.from_template(
    """
    Eres un Agente de Viajes experto.
    Tu tarea es crear un itinerario de ALTO NIVEL para el siguiente viaje:
    "{request}"
    
    Genera una lista numerada de pasos. NO entres en detalles de precios aún.
    Solo los pasos lógicos.
    Ejemplo:
    1. Buscar vuelos a París.
    2. Buscar hotel en el centro.
    3. Buscar entradas para el Louvre.
    """
)

planner_chain = planner_prompt | llm | StrOutputParser()

# --- 2. EL EJECUTOR (EXECUTOR) ---
# Simulamos herramientas para no complicar el ejemplo con APIs reales
def search_flights(destination):
    return f"✈️ Vuelos encontrados a {destination}: $500 (Iberia)"

def search_hotels(location):
    return f"🏨 Hotel en {location}: 'Grand Hotel' ($120/noche)"

def search_activities(activity):
    return f"🎟️ Tickets para {activity}: Disponibles para mañana."

executor_prompt = ChatPromptTemplate.from_template(
    """
    Eres un asistente de ejecución.
    Tu tarea es ejecutar el siguiente paso del plan: "{step}"
    
    Usa tu conocimiento o simula que usas una herramienta para dar un resultado.
    Responde SOLO con el resultado.
    """
)

executor_chain = executor_prompt | llm | StrOutputParser()

# --- 3. ORQUESTACIÓN ---
def run_plan_and_solve(request):
    print(f"🌍 SOLICITUD DE VIAJE: {request}\n")
    
    # FASE 1: PLANIFICACIÓN
    print("📋 GENERANDO PLAN MAESTRO...")
    plan_text = planner_chain.invoke({"request": request})
    print(f"{plan_text}\n")
    
    # Convertir el texto del plan en una lista (simple split por saltos de línea)
    steps = [line for line in plan_text.split('\n') if line.strip() and line[0].isdigit()]
    
    # FASE 2: EJECUCIÓN
    results = []
    print("⚙️ EJECUTANDO PLAN...")
    for step in steps:
        print(f"   👉 Ejecutando: {step}")
        # Aquí podríamos tener un router que decida qué herramienta real usar.
        # Para simplificar, le pedimos al LLM que simule la ejecución o usemos lógica simple.
        
        # Simulación simple basada en palabras clave
        if "vuelo" in step.lower():
            result = search_flights("destino del plan")
        elif "hotel" in step.lower():
            result = search_hotels("destino del plan")
        else:
            # Fallback al LLM para pasos genéricos
            result = executor_chain.invoke({"step": step})
            
        print(f"      ✅ Hecho: {result}")
        results.append(result)
        
    print("\n🎉 VIAJE PLANIFICADO CON ÉXITO.")
    return results

if __name__ == "__main__":
    run_plan_and_solve("Quiero un viaje de 3 días a Roma para ver el Coliseo y comer pizza.")
