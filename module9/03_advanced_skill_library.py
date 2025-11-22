"""
Módulo 9 - Ejemplo Avanzado: Biblioteca de Habilidades (Skill Library)
Framework: LangChain + ChromaDB
Caso de uso: Agente que construye su propia biblioteca de funciones reutilizables

El agente aprende nuevas habilidades (funciones Python), las almacena en un vector store
y las reutiliza en tareas futuras.

Instalación:
pip install langchain langchain-openai chromadb
"""

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import json

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
EMBEDDINGS = OpenAIEmbeddings()


class SkillLibrary:
    """Biblioteca de habilidades reutilizables"""
    
    def __init__(self, persist_directory: str = "./skill_library_db"):
        self.llm = LLM
        self.persist_directory = persist_directory
        
        # Cargar o crear vectorstore
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=EMBEDDINGS,
                collection_name="skills"
            )
            print(f"✅ Biblioteca cargada: {self.vectorstore._collection.count()} habilidades")
        except:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=EMBEDDINGS,
                collection_name="skills"
            )
            print("📚 Nueva biblioteca creada")
    
    def learn_skill(self, task_description: str, solution_code: str):
        """Aprender una nueva habilidad"""
        print(f"\n📖 Aprendiendo nueva habilidad...")
        
        # Generar descripción semántica
        desc_prompt = f"""Analiza esta función y crea una descripción semántica útil para búsqueda futura.

Función:
```python
{solution_code}
```

Tarea que resuelve: {task_description}

Crea una descripción que incluya:
1. Qué hace la función (propósito) - 1 línea
2. Casos de uso (cuándo usarla) - 1 línea
3. Keywords clave (tecnologías, conceptos) - lista

Formato JSON:
{{
    "purpose": "...",
    "use_cases": "...",
    "keywords": ["...", "..."]
}}"""
        
        response = self.llm.invoke(desc_prompt)
        
        try:
            metadata = json.loads(response.content)
        except:
            metadata = {
                "purpose": task_description,
                "use_cases": "General purpose",
                "keywords": []
            }
        
        # Crear documento
        full_description = f"""
Propósito: {metadata['purpose']}
Casos de uso: {metadata['use_cases']}
Keywords: {', '.join(metadata['keywords'])}

Código:
{solution_code}
"""
        
        doc = Document(
            page_content=full_description,
            metadata={
                "task": task_description,
                "code": solution_code,
                **metadata
            }
        )
        
        # Agregar a vectorstore
        self.vectorstore.add_documents([doc])
        
        print(f"   ✅ Habilidad aprendida: {metadata['purpose'][:60]}...")
    
    def search_skills(self, task: str, k: int = 3) -> List[Dict]:
        """Buscar habilidades relevantes para una tarea"""
        print(f"\n🔍 Buscando habilidades relevantes para: '{task}'...")
        
        results = self.vectorstore.similarity_search(task, k=k)
        
        skills = []
        for i, doc in enumerate(results, 1):
            skill = {
                "purpose": doc.metadata.get("purpose", "N/A"),
                "code": doc.metadata.get("code", ""),
            }
            skills.append(skill)
            print(f"   {i}. {skill['purpose'][:70]}...")
        
        return skills
    
    def solve_with_skills(self, task: str) -> str:
        """Resolver tarea usando habilidades existentes"""
        print(f"\n🎯 Resolviendo tarea: {task}")
        
        # Buscar habilidades relevantes
        relevant_skills = self.search_skills(task, k=3)
        
        if not relevant_skills:
            print("   No hay habilidades relevantes, creando desde cero...")
            return self._create_new_solution(task)
        
        # Usar habilidades existentes
        skills_context = "\n\n".join([
            f"Habilidad {i+1}:\n```python\n{skill['code']}\n```"
            for i, skill in enumerate(relevant_skills)
        ])
        
        solve_prompt = f"""Tarea: {task}

Habilidades disponibles en tu biblioteca:
{skills_context}

REUTILIZA y ADAPTA estas habilidades existentes para resolver la nueva tarea.
No reinventes la rueda si ya tienes código similar.

Escribe la solución COMPLETA (solo código Python):"""
        
        response = self.llm.invoke(solve_prompt)
        code = response.content.strip()
        
        # Limpiar
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        print(f"\n✅ Solución generada reutilizando habilidades")
        
        return code
    
    def _create_new_solution(self, task: str) -> str:
        """Crear solución desde cero"""
        prompt = f"Tarea: {task}\n\nEscribe código Python completo (solo código):"
        response = self.llm.invoke(prompt)
        code = response.content.strip()
        
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        
        return code


def main():
    """Demostración de Skill Library"""
    print("=" * 80)
    print("Sistema de Biblioteca de Habilidades - Aprendizaje Continuo")
    print("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    # Crear biblioteca
    library = SkillLibrary()
    
    # FASE 1: Aprender habilidades básicas
    print("\n" + "="*80)
    print("FASE 1: APRENDIZAJE - Construyendo biblioteca")
    print("="*80)
    
    initial_skills = [
        {
            "task": "Validar si un email es válido usando regex",
            "code": """import re

def validate_email(email: str) -> bool:
    \"\"\"Valida si un email tiene formato correcto.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))"""
        },
        {
            "task": "Convertir JSON a CSV",
            "code": """import json
import csv

def json_to_csv(json_data: list, output_file: str):
    \"\"\"Convierte lista de dicts JSON a archivo CSV.\"\"\"
    if not json_data:
        return
    
    keys = json_data[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(json_data)"""
        },
        {
            "task": "Calcular hash SHA256 de un archivo",
            "code": """import hashlib

def file_hash(filepath: str) -> str:
    \"\"\"Calcula SHA256 hash de un archivo.\"\"\"
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()"""
        }
    ]
    
    for skill in initial_skills:
        library.learn_skill(skill["task"], skill["code"])
    
    # FASE 2: Resolver nuevas tareas reutilizando habilidades
    print("\n" + "="*80)
    print("FASE 2: APLICACIÓN - Resolviendo nuevas tareas con habilidades")
    print("="*80)
    
    new_tasks = [
        "Validar una lista de emails y filtrar solo los válidos",
        "Leer un archivo JSON y exportarlo a CSV",
    ]
    
    for task in new_tasks:
        print(f"\n{'-'*80}")
        solution = library.solve_with_skills(task)
        print(f"\n📝 Solución:\n```python\n{solution}\n```")
        
        # Aprender esta nueva solución también
        library.learn_skill(task, solution)
    
    # FASE 3: Demostrar el crecimiento
    print("\n" + "="*80)
    print("FASE 3: EVOLUCIÓN - Estado de la biblioteca")
    print("="*80)
    
    total_skills = library.vectorstore._collection.count()
    print(f"\n🧠 Total de habilidades en biblioteca: {total_skills}")
    print(f"\n💡 El agente puede ahora resolver nuevas tareas más rápidamente")
    print("   reutilizando y combinando habilidades existentes.")
    
    # Demostrar búsqueda semántica
    print(f"\n🔍 Búsqueda semántica demo:")
    query = "necesito procesar archivos para seguridad"
    library.search_skills(query)


if __name__ == "__main__":
    main()
