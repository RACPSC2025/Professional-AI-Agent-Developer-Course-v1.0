"""
02_tree_of_thoughts.py
======================
Implementación del algoritmo Tree of Thoughts (ToT) para resolución de problemas complejos.
ToT permite al modelo explorar múltiples ramas de razonamiento, evaluar su promesa y podar las malas.

Problema ejemplo: Creative Writing (Escribir una historia coherente de 3 oraciones)
Algoritmo: BFS (Breadth-First Search)

Requisitos:
pip install langchain langchain-openai
"""

import operator
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Configuración
llm = ChatOpenAI(model="gpt-4", temperature=0.7) # GPT-4 recomendado para razonamiento

class TreeOfThoughts:
    def __init__(self, k=3, b=5):
        self.k = k # Número de pensamientos a generar por paso (branching factor)
        self.b = b # Número de mejores pensamientos a mantener (beam width)
        
    def generate_thoughts(self, current_state: str, step: int) -> List[str]:
        """Genera k posibles continuaciones"""
        prompt = f"""
        Estamos escribiendo una historia de 3 oraciones.
        Estado actual: "{current_state}"
        Paso actual: {step}/3
        
        Genera {self.k} posibles siguientes oraciones que continúen la historia de forma coherente e interesante.
        Devuelve solo las oraciones, una por línea.
        """
        response = llm.invoke(prompt).content
        thoughts = [t.strip() for t in response.split('\n') if t.strip()]
        return thoughts[:self.k]

    def evaluate_thoughts(self, state: str, thoughts: List[str]) -> List[float]:
        """Evalúa la calidad de cada pensamiento (0.0 a 1.0)"""
        scores = []
        for thought in thoughts:
            prompt = f"""
            Evalúa la siguiente continuación para la historia:
            Historia previa: "{state}"
            Continuación propuesta: "{thought}"
            
            Asigna un puntaje de 0.0 a 1.0 basado en coherencia y creatividad.
            Devuelve SOLO el número.
            """
            try:
                score = float(llm.invoke(prompt).content.strip())
            except:
                score = 0.5
            scores.append(score)
        return scores

    def solve(self, initial_prompt: str, steps: int = 3):
        current_states = [initial_prompt] # Lista de estados prometedores
        
        print(f"🌳 Iniciando Tree of Thoughts para: '{initial_prompt}'\n")
        
        for step in range(1, steps + 1):
            print(f"--- Paso {step} ---")
            candidates = []
            
            # 1. Expand (Generar pensamientos para cada estado actual)
            for state in current_states:
                thoughts = self.generate_thoughts(state, step)
                
                # 2. Evaluate (Puntuar pensamientos)
                scores = self.evaluate_thoughts(state, thoughts)
                
                for t, s in zip(thoughts, scores):
                    new_state = f"{state} {t}"
                    candidates.append({'state': new_state, 'score': s, 'thought': t})
                    print(f"   💡 Idea: ...{t[:30]}... | Score: {s}")
            
            # 3. Prune (Seleccionar los top b mejores globales)
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidates = candidates[:self.b]
            
            current_states = [c['state'] for c in best_candidates]
            print(f"   ✂️ Podando... Mantenemos {len(current_states)} ramas.")
            print(f"   🌟 Mejor actual: {current_states[0]}\n")
            
        return current_states[0]

def main():
    tot = TreeOfThoughts(k=3, b=2)
    initial = "El detective encontró un reloj roto en el suelo."
    
    final_story = tot.solve(initial, steps=3)
    
    print("📖 Historia Final Generada por ToT:")
    print(final_story)

if __name__ == "__main__":
    main()
