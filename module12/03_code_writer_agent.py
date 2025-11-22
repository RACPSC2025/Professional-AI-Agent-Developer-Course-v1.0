"""
Módulo 12 - Proyecto Capstone: Code Writer Agent
Framework: LangChain + Execution Sand box
Parte 3: Agente que escribe código y tests

Genera código basado en análisis, valida sintaxis, escribe tests y crea PRs.

Instalación:
pip install langchain langchain-openai PyGithub
"""

import os
from typing import Dict, List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o", temperature=0.3)


class CodeWriterAgent:
    """Agente que escribe código y tests"""
    
    def __init__(self):
        self.llm = LLM
        print("✍️ Code Writer Agent inicializado")
    
    def generate_code_fix(self, issue: str, analysis: str, language: str = "python") -> str:
        """Generar código para solucionar un issue"""
        print(f"\n💻 Generando código ({language})...")
        
        code_prompt = f"""Eres un senior developer escribiendo código production-ready.

Issue a resolver: {issue}

Análisis técnico:
{analysis}

Genera código {language} que:
1. Solucione el issue completamente
2. Siga best practices y PEP 8 (si es Python)
3. Incluya docstrings y comentarios donde sea necesario
4. Maneje edge cases y errores
5. Sea modular y testeable

Responde SOLO con el código, sin explicaciones adicionales."""
        
        response = self.llm.invoke(code_prompt)
        code = response.content
        
        # Limpiar markdown si existe
        if f"```{language}" in code:
            code = code.split(f"```{language}")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        print(f"   ✅ Código generado ({len(code.split(chr(10)))} líneas)")
        
        return code
    
    def generate_tests(self, code: str, language: str = "python") -> str:
        """Generar tests unitarios para el código"""
        print(f"\n🧪 Generando tests...")
        
        test_prompt = f"""Genera tests unitarios comprehensivos para este código:

```{language}
{code}
```

Tests deben:
1. Cubrir casos normales
2. Cubrir edge cases
3. Test errores y excepciones
4. Usar assertions claros
5. Ser independientes entre sí

Framework: pytest (Python) / jest (JavaScript) según el lenguaje.

Responde SOLO con el código de tests."""
        
        response = self.llm.invoke(test_prompt)
        tests = response.content
        
        # Limpiar
        if "```" in tests:
            tests = tests.split("```")[1].split("```")[0].strip()
            if tests.startswith("python") or tests.startswith("javascript"):
                tests = tests.split('\n', 1)[1].strip()
        
        print(f"   ✅ Tests generados ({len(tests.split(chr(10)))} líneas)")
        
        return tests
    
    def validate_code(self, code: str, language: str = "python") -> Dict:
        """Validar código (sintaxis, estilo, lógica)"""
        print(f"\n✅ Validando código...")
        
        issues = []
        
        # Validación de sintaxis (solo Python en este ejemplo)
        if language == "python":
            try:
                import ast
                ast.parse(code)
                print("   ✅ Sintaxis válida")
            except SyntaxError as e:
                issues.append({
                    "type": "syntax_error",
                    "severity": "critical",
                    "message": f"Syntax error: {str(e)}"
                })
                print(f"   ❌ Error de sintaxis: {e}")
        
        # Validación con LLM (lógica, buenas prácticas)
        validation_prompt = f"""Revisa este código críticamente:

```{language}
{code[:1500]}  # Limitar
```

Identifica:
1. Errores lógicos
2. Violaciones de best practices
3. Problemas de seguridad
4. Performance issues

Si el código es correcto, responde: "APROBADO"
Si hay problemas, lista máximo 3 problemas críticos."""
        
        response = self.llm.invoke(validation_prompt)
        validation_result = response.content
        
        if "APROBADO" not in validation_result:
            issues.append({
                "type": "quality_issues",
                "severity": "medium",
                "message": validation_result[:200]
            })
            print(f"   ⚠️ Problemas de calidad detectados")
        else:
            print("   ✅ Código aprobado")
        
        return {
            "is_valid": len([i for i in issues if i['severity'] == 'critical']) == 0,
            "issues": issues
        }
    
    def create_pull_request_description(self, issue_number: int, code: str, tests: str) -> str:
        """Crear descripción de PR automáticamente"""
        print(f"\n📝 Generando descripción de PR...")
        
        pr_prompt = f"""Crea una descripción profesional de Pull Request para GitHub:

Issue #: {issue_number}

Código implementado:
{code[:500]}...

Tests:
{tests[:300]}...

La descripción debe incluir:
## Descripción
[Qué hace este PR]

## Cambios
- [Lista de cambios principales]

## Testing
- [Cómo se testeó]

## Checklist
- [ ] Tests pasando
- [ ] Código revisado
- [ ] Documentación actualizada

Formato Markdown profesional."""
        
        response = self.llm.invoke(pr_prompt)
        pr_description = response.content
        
        print("   ✅ Descripción de PR generada")
        
        return pr_description
    
    def generate_commit_message(self, issue_number: int, changes_summary: str) -> str:
        """Generar mensaje de commit semántico"""
        commit_prompt = f"""Genera un commit message siguiendo Conventional Commits:

Issue #{issue_number}
Cambios: {changes_summary}

Formato:
type(scope): subject

body (opcional)

Closes #{issue_number}

Donde type = fix|feat|docs|style|refactor|test|chore"""
        
        response = self.llm.invoke(commit_prompt)
        commit_msg = response.content.strip()
        
        return commit_msg


def main():
    """Demostración del Code Writer Agent"""
    print("=" * 70)
    print("Code Writer Agent - Capstone Project")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    # Crear agente
    agent = CodeWriterAgent()
    
    # Caso de ejemplo
    issue_desc = "Fix SQL injection vulnerability in login function"
    analysis = """
    Root cause: User input concatenated directly in SQL query
    Solution: Use parameterized queries
    Files affected: auth.py
    """
    
    print(f"\n{'=' * 70}")
    print("WORKFLOW DE CÓDIGO")
    print('=' * 70)
    
    # 1. Generar código fix
    fixed_code = agent.generate_code_fix(issue_desc, analysis, "python")
    
    print(f"\n📄 Código generado:")
    print(f"```python\n{fixed_code}\n```")
    
    # 2. Generar tests
    tests = agent.generate_tests(fixed_code, "python")
    
    print(f"\n📄 Tests generados:")
    print(f"```python\n{tests}\n```")
    
    # 3. Validar
    validation = agent.validate_code(fixed_code, "python")
    
    if validation['is_valid']:
        print(f"\n✅ Validación exitosa")
    else:
        print(f"\n⚠️ Problemas encontrados:")
        for issue in validation['issues']:
            print(f"   - {issue['type']}: {issue['message'][:100]}")
    
    # 4. Crear PR description
    pr_desc = agent.create_pull_request_description(123, fixed_code, tests)
    
    print(f"\n{'=' * 70}")
    print("PULL REQUEST DESCRIPTION")
    print('=' * 70)
    print(pr_desc)
    
    # 5. Commit message
    commit_msg = agent.generate_commit_message(123, "Fix SQL injection in auth")
    
    print(f"\n{'=' * 70}")
    print("COMMIT MESSAGE")
    print('=' * 70)
    print(commit_msg)
    
    print(f"\n\n💡 Este agente puede:")
    print("   ✅ Generar código production-ready")
    print("   ✅ Escribir tests comprehensivos")
    print("   ✅ Validar sintaxis y calidad")
    print("   ✅ Crear PRs con descripciones profesionales")
    print("   ✅ Seguir convenciones (Conventional Commits, etc.)")


if __name__ == "__main__":
    main()
