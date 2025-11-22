"""
🟢 NIVEL BÁSICO: COMPARATIVA DE SINTAXIS (THE ROSETTA STONE)
------------------------------------------------------------
Este script resuelve el MISMO problema usando dos enfoques distintos:
1. LangChain (Enfoque Abstracto/Modular).
2. Google GenAI SDK (Enfoque Nativo/Directo).

El Problema: Generar un perfil de personaje de RPG en formato JSON estricto.
Esto demuestra cómo cada framework maneja "Structured Output".

Requisitos:
pip install langchain-openai google-generativeai pydantic python-dotenv
"""

import os
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Google Imports
import google.generativeai as genai

load_dotenv()

# --- DEFINICIÓN DEL ESQUEMA (MODELO DE DATOS) ---
class RPGCharacter(BaseModel):
    name: str = Field(description="Nombre del personaje")
    role: str = Field(description="Clase o rol (ej. Mago, Guerrero)")
    stats: dict = Field(description="Diccionario con stats: STR, DEX, INT")
    inventory: List[str] = Field(description="Lista de 3 items iniciales")

# ==========================================
# ENFOQUE 1: LANGCHAIN (LCEL)
# ==========================================
def run_langchain_version(theme: str):
    print("\n🔵 EJECUTANDO VERSIÓN LANGCHAIN...")
    
    # 1. Modelo
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 2. Parser
    parser = PydanticOutputParser(pydantic_object=RPGCharacter)
    
    # 3. Prompt con instrucciones de formato inyectadas
    prompt = ChatPromptTemplate.from_template(
        """Genera un personaje de RPG basado en el tema: {theme}.
        {format_instructions}
        """
    )
    
    # 4. Chain
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "theme": theme,
            "format_instructions": parser.get_format_instructions()
        })
        print(f"✅ LangChain Resultado (Objeto Pydantic):\n{result}")
        print(f"   Nombre: {result.name}, Rol: {result.role}")
    except Exception as e:
        print(f"❌ Error en LangChain: {e}")

# ==========================================
# ENFOQUE 2: GOOGLE GENAI SDK (NATIVO)
# ==========================================
def run_google_version(theme: str):
    print("\n🟣 EJECUTANDO VERSIÓN GOOGLE SDK...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Salta Google Version: No GOOGLE_API_KEY encontrada.")
        return

    genai.configure(api_key=api_key)
    
    # 1. Configuración del Modelo con "Response Schema" (Nativo en Gemini 1.5)
    # Nota: Gemini Pro 1.5 soporta JSON mode nativo
    generation_config = {
        "temperature": 0.7,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    # 2. Prompt
    prompt_text = f"""
    Genera un personaje de RPG basado en el tema: {theme}.
    Sigue este esquema JSON:
    {{
        "name": "string",
        "role": "string",
        "stats": {{"STR": int, "DEX": int, "INT": int}},
        "inventory": ["string", "string", "string"]
    }}
    """
    
    try:
        response = model.generate_content(prompt_text)
        # Parsear el JSON raw
        data = json.loads(response.text)
        print(f"✅ Google Resultado (Diccionario):\n{json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"❌ Error en Google SDK: {e}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("--- ⚔️ RPG CHARACTER GENERATOR BATTLE ⚔️ ---")
    user_theme = input("Elige un tema (ej. Cyberpunk, Fantasía Medieval, Espacial): ") or "Cyberpunk"
    
    run_langchain_version(user_theme)
    run_google_version(user_theme)
