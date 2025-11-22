"""
🟢 NIVEL BÁSICO: RESUMIDOR DE NOTICIAS (THE CHAIN)
--------------------------------------------------
Este script demuestra una "Cadena" (Chain) secuencial usando LCEL (LangChain Expression Language).

❌ POR QUÉ NO ES UN AGENTE:
   - No tiene autonomía: No puede decidir qué hacer.
   - Flujo rígido: Siempre ejecuta A -> B -> C.
   - Sin bucle: No puede reintentar si algo falla ni buscar más información.

✅ PARA QUÉ SIRVE:
   - Tareas repetitivas y predecibles (ETL, Resúmenes, Traducción).

Conceptos Clave:
- LCEL: La sintaxis `|` para componer flujos de datos.
- PromptTemplates: Plantillas reutilizables para instrucciones al LLM.
"""

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cargar claves de API
load_dotenv()

# 1. Definir el componente de Scraping (Función Python pura)
def scrape_website(url: str):
    """Descarga el contenido de texto de una web."""
    print(f"🕷️  Scrapeando: {url}...")
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "html.parser")
        # Extraer solo párrafos para limpiar ruido
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text[:10000] # Limitar a 10k caracteres para no saturar contexto
    except Exception as e:
        return f"Error scrapeando: {e}"

# 2. Definir el Prompt
summary_prompt = ChatPromptTemplate.from_template(
    """
    Actúa como un editor de noticias tech experto.
    Resume el siguiente texto en 3 puntos clave (bullet points).
    Usa emojis para hacerlo visual.
    
    TEXTO ORIGINAL:
    {text}
    
    RESUMEN:
    """
)

# 3. Inicializar el Modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Construir la Cadena (The Chain) usando LCEL
# Flujo: Input (URL) -> scrape_website -> prompt -> llm -> string_parser
chain = (
    {"text": scrape_website} 
    | summary_prompt 
    | llm 
    | StrOutputParser()
)

# 5. Ejecución
if __name__ == "__main__":
    print("--- 📰 AI NEWS SUMMARIZER ---")
    url = input("Introduce la URL de un artículo: ")
    
    if url:
        print("\n⏳ Procesando...")
        result = chain.invoke(url)
        print("\n--- 📝 RESUMEN ---")
        print(result)
