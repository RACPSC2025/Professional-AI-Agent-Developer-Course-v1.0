"""
Módulo 0 - Framework Decision Tree
Selector inteligente de framework basado en requisitos del proyecto

Responde preguntas y recomienda el framework más apropiado.

Instalación:
pip install langchain langchain-openai python-dotenv
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class FrameworkSelector:
    """Ayuda a elegir el framework correcto para tu proyecto"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    def analyze_requirements(self, answers: dict) -> dict:
        """Analizar requisitos y recomendar framework"""
        
        # Decision logic
        score = {
            "langchain": 0,
            "langgraph": 0,
            "crewai": 0,
            "autogen": 0,
            "semantic_kernel": 0
        }
        
        # Pregunta 1: Single vs Multi-agent
        if answers["agent_count"] == "single":
            score["langchain"] += 3
            score["langgraph"] += 2
        else:
            score["crewai"] += 4
            score["autogen"] += 3
            score["langgraph"] += 2
        
        # Pregunta 2: Complejidad
        if answers["complexity"] == "simple":
            score["langchain"] += 3
            score["crewai"] += 2
        elif answers["complexity"] == "medium":
            score["langchain"] += 2
            score["langgraph"] += 2
            score["autogen"] += 2
        else:  # complex
            score["langgraph"] += 4
            score["autogen"] += 2
        
        # Pregunta 3: Tipo de workflow
        if answers["workflow_type"] == "linear":
            score["crewai"] += 3
            score["langchain"] += 2
        elif answers["workflow_type"] == "branching":
            score["langgraph"] += 5
        elif answers["workflow_type"] == "conversational":
            score["autogen"] += 5
        
        # Pregunta 4: Prioridad
        if answers["priority"] == "speed":
            score["crewai"] += 2
            score["langchain"] += 2
        elif answers["priority"] == "control":
            score["langgraph"] += 4
        elif answers["priority"] == "simplicity":
            score["crewai"] += 3
            score["langchain"] += 2
        
        # Pregunta 5: Ecosistema
        if answers["ecosystem"] == "microsoft":
            score["semantic_kernel"] += 5
        else:
            score["langchain"] += 1
        
        # Encontrar ganador
        winner = max(score, key=score.get)
        
        return {
            "recommended": winner,
            "scores": score,
            "confidence": score[winner] / sum(score.values())
        }
    
    def get_justification(self, winner: str, answers: dict) -> str:
        """Generar justificación usando LLM"""
        
        justifications = {
            "langchain": "General purpose perfecto para RAG y prototipos",
            "langgraph": "Control explícito ideal para workflows complejos",
            "crewai": "Simplicidad y rapidez en multi-agent teams",
            "autogen": "Excelente para conversaciones y debates",
            "semantic_kernel": "Mejor integración con ecosistema Microsoft"
        }
        
        prompt = f"""Basándote en estos requisitos del proyecto:

Número de agentes: {answers['agent_count']}
Complejidad: {answers['complexity']}
Tipo de workflow: {answers['workflow_type']}
Prioridad: {answers['priority']}
Ecosistema: {answers['ecosystem']}

Recomienda framework: {winner}

Justifica en 2-3 líneas por qué {winner} es la mejor opción.
Menciona pros y cons específicos."""
        
        response = self.llm.invoke(prompt)
        return response.content


def interactive_questionnaire():
    """Cuestionario interactivo para el usuario"""
    
    print("="*80)
    print("FRAMEWORK DECISION TREE")
    print("="*80)
    print("\nResponde estas preguntas para obtener una recomendación:\n")
    
    answers = {}
    
    # Pregunta 1
    print("1. ¿Cuántos agentes necesitas?")
    print("   a) Un solo agente")
    print("   b) Múltiples agentes")
    choice = input("   Respuesta (a/b): ").lower()
    answers["agent_count"] = "single" if choice == "a" else "multi"
    
    # Pregunta 2
    print("\n2. ¿Qué tan complejo es tu workflow?")
    print("   a) Simple (linear, pocas decisiones)")
    print("   b) Medio (algunas decisiones, iteraciones)")
    print("   c) Complejo (muchas ramas, loops, estado)")
    choice = input("   Respuesta (a/b/c): ").lower()
    complexity_map = {"a": "simple", "b": "medium", "c": "complex"}
    answers["complexity"] = complexity_map.get(choice, "medium")
    
    # Pregunta 3
    print("\n3. ¿Qué tipo de workflow es?")
    print("   a) Lineal/secuencial (paso a paso)")
    print("   b) Con decisiones/branches condicionales")
    print("   c) Conversacional (agentes debaten/discuten)")
    choice = input("   Respuesta (a/b/c): ").lower()
    workflow_map = {"a": "linear", "b": "branching", "c": "conversational"}
    answers["workflow_type"] = workflow_map.get(choice, "linear")
    
    # Pregunta 4
    print("\n4. ¿Qué es más importante para ti?")
    print("   a) Velocidad de desarrollo (prototipo rápido)")
    print("   b) Control total sobre el flujo")
    print("   c) Simplicidad del código")
    choice = input("   Respuesta (a/b/c): ").lower()
    priority_map = {"a": "speed", "b": "control", "c": "simplicity"}
    answers["priority"] = priority_map.get(choice, "simplicity")
    
    # Pregunta 5
    print("\n5. ¿Qué ecosistema usas?")
    print("   a) Python/Open Source")
    print("   b) Microsoft/.NET")
    choice = input("   Respuesta (a/b): ").lower()
    answers["ecosystem"] = "microsoft" if choice == "b" else "python"
    
    return answers


def main():
    """Demo del selector"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY no configurada")
        return
    
    # Modo interactivo
    use_interactive = input("\n¿Usar modo interactivo? (s/n): ").lower() == "s"
    
    if use_interactive:
        answers = interactive_questionnaire()
    else:
        # Ejemplo predefinido
        print("\nUsando ejemplo predefinido...")
        answers = {
            "agent_count": "multi",
            "complexity": "complex",
            "workflow_type": "branching",
            "priority": "control",
            "ecosystem": "python"
        }
        
        print("\nRequisitos del proyecto:")
        for key, value in answers.items():
            print(f"  - {key}: {value}")
    
    # Analizar
    selector = FrameworkSelector()
    result = selector.analyze_requirements(answers)
    
    # Mostrar resultado
    print("\n" + "="*80)
    print("RECOMENDACIÓN")
    print("="*80)
    
    print(f"\n🎯 Framework Recomendado: {result['recommended'].upper()}")
    print(f"   Confianza: {result['confidence']*100:.0f}%")
    
    print("\n📊 Scores de todos los frameworks:")
    for framework, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score)
        print(f"   {framework:20} {score:2} {bar}")
    
    # Justificación
    justification = selector.get_justification(result['recommended'], answers)
    print(f"\n💡 Justificación:")
    print(f"   {justification}")
    
    # Quick start
    print(f"\n🚀 Quick Start con {result['recommended']}:")
    
    quickstarts = {
        "langchain": "pip install langchain langchain-openai",
        "langgraph": "pip install langgraph langchain-openai",
        "crewai": "pip install crewai",
        "autogen": "pip install pyautogen",
        "semantic_kernel": "pip install semantic-kernel"
    }
    
    print(f"   {quickstarts[result['recommended']]}")
    
    print("\n📚 Recursos:")
    docs = {
        "langchain": "https://python.langchain.com",
        "langgraph": "https://langchain-ai.github.io/langgraph/",
        "crewai": "https://docs.crewai.com",
        "autogen": "https://microsoft.github.io/autogen/",
        "semantic_kernel": "https://learn.microsoft.com/semantic-kernel"
    }
    print(f"   Docs: {docs[result['recommended']]}")


if __name__ == "__main__":
    main()
