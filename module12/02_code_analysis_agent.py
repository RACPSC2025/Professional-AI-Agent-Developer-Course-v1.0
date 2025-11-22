"""
Módulo 12 - Proyecto Capstone: Code Analysis Agent
Framework: LangChain + AST Analysis
Parte 2: Agente especializado en análisis de código

Analiza código, detecta bugs, identifica root causes y sugiere soluciones.

Instalación:
pip install langchain langchain-openai ast-grep-py
"""

import os
import ast
from typing import List, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o", temperature=0.2)  # Baja temp para consistencia


class CodeAnalysisAgent:
    """Agente especializado en análisis de código"""
    
    def __init__(self):
        self.llm = LLM
        print("🔬 Code Analysis Agent inicializado")
    
    def analyze_issue(self, issue_description: str, codebase_context: str = "") -> Dict:
        """Analizar issue y proporcionar análisis técnico"""
        print(f"\n🔍 Analizando issue...")
        print(f"   Descripción: {issue_description[:100]}...")
        
        analysis_prompt = f"""Eres un senior developer analizando un bug report.

Issue: {issue_description}

Contexto del codebase:
{codebase_context if codebase_context else "No additional context provided"}

Proporciona un análisis técnico detallado:

1. **ROOT CAUSE**: ¿Cuál es la causa raíz probable del problema?
2. **AFFECTED FILES**: ¿Qué archivos probablemente necesitan cambios?
3. **APPROACH**: ¿Cuál es el mejor approach para solucionarlo?
4. **RISKS**: ¿Qué riesgos o side effects debemos considerar?
5. **TESTS NEEDED**: ¿Qué tests se necesitan?

Sé específico y técnico."""
        
        response = self.llm.invoke(analysis_prompt)
        analysis = response.content
        
        print(f"\n✅ Análisis completado")
        
        return {
            "analysis": analysis,
            "confidence": 0.85  # Simplificado
        }
    
    def detect_code_smells(self, code: str, language: str = "python") -> List[Dict]:
        """Detectar code smells y anti-patterns"""
        print(f"\n👃 Detectando code smells en código {language}...")
        
        smells = []
        
        if language == "python":
            try:
                # Analizar con AST
                tree = ast.parse(code)
                
                # Detector 1: Funciones muy largas
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        lines = len(ast.unparse(node).split('\n'))
                        if lines > 50:
                            smells.append({
                                "type": "long_function",
                                "severity": "medium",
                                "location": f"Function {node.name}",
                                "message": f"Function is {lines} lines (recommended < 50)"
                            })
                
                # Detector 2: Exceso de parámetros
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if len(node.args.args) > 5:
                            smells.append({
                                "type": "too_many_parameters",
                                "severity": "low",
                                "location": f"Function {node.name}",
                                "message": f"Has {len(node.args.args)} parameters (recommended ≤ 5)"
                            })
                
            except SyntaxError:
                print("   ⚠️ Could not parse code (syntax error)")
        
        # Análisis con LLM para patrones más complejos
        llm_analysis_prompt = f"""Analiza este código y detecta problemas:

```{language}
{code[:1000]}  # Limitar tamaño
```

Identifica:
- Vulnerabilidades de seguridad
- Performance issues
- Violaciones de principios SOLID
- Magic numbers sin constantes
- Falta de manejo de errores

Lista solo problemas reales (máximo 3)."""
        
        llm_response = self.llm.invoke(llm_analysis_prompt)
        
        # Parsear respuesta del LLM (simplificado)
        if "no encontr" not in llm_response.content.lower():
            smells.append({
                "type": "llm_detected",
                "severity": "review",
                "location": "General",
                "message": llm_response.content[:200]
            })
        
        print(f"   Encontrados {len(smells)} posibles problemas")
        
        return smells
    
    def suggest_fixes(self, issue: str, analysis: str) -> List[str]:
        """Sugerir soluciones específicas"""
        print(f"\n💡 Generando sugerencias de solución...")
        
        suggestion_prompt = f"""Basándote en el análisis:

Issue: {issue}

Análisis:
{analysis}

Propón 3 soluciones concretas, ordenadas de mejor a peor:
1. [Solución preferida]
2. [Alternativa]
3. [Último recurso]

Sé específico sobre QUÉ cambios hacer."""
        
        response = self.llm.invoke(suggestion_prompt)
        suggestions_text = response.content
        
        # Parsear en lista (simplificado)
        suggestions = [
            line.strip() for line in suggestions_text.split('\n')
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))
        ]
        
        print(f"   ✅ Generadas {len(suggestions)} sugerencias")
        
        return suggestions
    
    def generate_analysis_report(self, issue_number: int, analysis_data: Dict) -> str:
        """Generar reporte estructurado de análisis"""
        report = f"# Code Analysis Report - Issue #{issue_number}\n\n"
        report += f"## 🔍 Technical Analysis\n\n"
        report += analysis_data.get('analysis', 'No analysis available')
        report += "\n\n"
        
        if 'code_smells' in analysis_data:
            report += f"## 👃 Code Quality Issues\n\n"
            for smell in analysis_data['code_smells']:
                report += f"- **{smell['type']}** ({smell['severity']}): "
                report += f"{smell['message']}\n"
            report += "\n"
        
        if 'suggestions' in analysis_data:
            report += f"## 💡 Recommended Solutions\n\n"
            for i, suggestion in enumerate(analysis_data['suggestions'], 1):
                report += f"{i}. {suggestion}\n"
            report += "\n"
        
        report += f"\n---\n*Analysis confidence: {analysis_data.get('confidence', 0)*100:.0f}%*\n"
        
        return report


def main():
    """Demostración del Code Analysis Agent"""
    print("=" * 70)
    print("Code Analysis Agent - Capstone Project")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    # Crear agente
    agent = CodeAnalysisAgent()
    
    # Caso de ejemplo
    issue_desc = """
    Bug Report: User login fails with 500 error
    
    Steps to reproduce:
    1. Go to /login
    2. Enter valid credentials
    3. Click login
    4. Server returns 500 error
    
    Expected: Successful login
    Actual: 500 Internal Server Error
    """
    
    example_code = """
def authenticate_user(username, password):
    user = database.query("SELECT * FROM users WHERE username = '" + username + "'")
    if user and user.password == password:
        return create_session(user)
    return None
"""
    
    print(f"\n{'=' * 70}")
    print("ANÁLISIS DE ISSUE")
    print('=' * 70)
    
    # 1. Analizar issue
    analysis_result = agent.analyze_issue(issue_desc, codebase_context=example_code)
    
    # 2. Detectar code smells
    code_smells = agent.detect_code_smells(example_code, "python")
    
    # 3. Sugerir fixes
    suggestions = agent.suggest_fixes(issue_desc, analysis_result['analysis'])
    
    # 4. Generar reporte
    full_analysis = {
        **analysis_result,
        "code_smells": code_smells,
        "suggestions": suggestions
    }
    
    report = agent.generate_analysis_report(123, full_analysis)
    
    print(f"\n{'=' * 70}")
    print("REPORTE FINAL")
    print('=' * 70)
    print(report)
    
    print("\n💡 Este agente puede:")
    print("   ✅ Analizar bugs y encontrar root causes")
    print("   ✅ Detectar code smells y malas prácticas")
    print("   ✅ Sugerir soluciones concretas")
    print("   ✅ Generar reportes estructurados")


if __name__ == "__main__":
    main()
