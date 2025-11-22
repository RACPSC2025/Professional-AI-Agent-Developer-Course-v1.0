"""
Módulo 12 - Proyecto Capstone: GitHub Agent Manager
Framework: CrewAI + LangGraph
Parte 1: Agente Manager que orquesta el equipo

Este agente recibe issues de GitHub, las analiza, las prioriza y las asigna
a los agentes especializados del equipo.

Instalación:
pip install crewai langchain langchain-openai PyGithub python-dotenv
"""

import os
from typing import List, Dict
from github import Github, GithubException
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class GitHubAgentManager:
    """Agente manager que gestiona el flujo de trabajo"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.gh = Github(github_token)
        self.repo = self.gh.get_repo(repo_name)
        self.llm = LLM
        
        print(f"✅ Manager conectado a repositorio: {repo_name}")
    
    def fetch_open_issues(self, limit: int = 10) -> List[Dict]:
        """Obtener issues abiertas del repositorio"""
        print(f"\n📋 Obteniendo issues abiertas...")
        
        issues = []
        for issue in self.repo.get_issues(state='open')[:limit]:
            issues.append({
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "labels": [label.name for label in issue.labels],
                "created_at": issue.created_at
            })
        
        print(f"   Encontradas {len(issues)} issues abiertas")
        return issues
    
    def triage_issue(self, issue: Dict) -> Dict:
        """Analizar y clasificar una issue"""
        print(f"\n🔍 Triaging Issue #{issue['number']}: {issue['title']}")
        
        triage_prompt = f"""Analiza esta GitHub issue y clasifícala:

Título: {issue['title']}
Descripción: {issue['body'][:500]}
Labels: {', '.join(issue['labels']) if issue['labels'] else 'ninguno'}

Proporciona:
1. TIPO: bug, feature, documentation, question, enhancement
2. PRIORIDAD: critical, high, medium, low
3. ASIGNAR_A: code_analyst, code_writer, documentation (elige el más apropiado)
4. COMPLEJIDAD: 1-10 (donde 10 es muy complejo)
5. RESUMEN: Resumen en 1 línea

Formato JSON:
{{
    "type": "...",
    "priority": "...",
    "assign_to": "...",
    "complexity": X,
    "summary": "..."
}}"""
        
        response = self.llm.invoke(triage_prompt)
        
        try:
            import json
            triage_result = json.loads(response.content)
        except:
            # Fallback
            triage_result = {
                "type": "unknown",
                "priority": "medium",
                "assign_to": "code_analyst",
                "complexity": 5,
                "summary": issue['title']
            }
        
        print(f"   Tipo: {triage_result['type']}")
        print(f"   Prioridad: {triage_result['priority']}")
        print(f"   Asignar a: {triage_result['assign_to']}")
        print(f"   Complejidad: {triage_result['complexity']}/10")
        
        return {**issue, **triage_result}
    
    def create_task_for_agent(self, triaged_issue: Dict) -> Dict:
        """Crear tarea específica para el agente asignado"""
        agent = triaged_issue['assign_to']
        
        task_templates = {
            "code_analyst": f"Analizar issue #{triaged_issue['number']} y identificar: "
                           f"1) Root cause, 2) Archivos afectados, 3) Approach sugerido",
            
            "code_writer": f"Implementar solución para issue #{triaged_issue['number']}: "
                          f"{triaged_issue['summary']}. Generar código y tests.",
            
            "documentation": f"Documentar solución/feature de issue #{triaged_issue['number']}. "
                            f"Actualizar README y docs relevantes."
        }
        
        task = {
            "issue_number": triaged_issue['number'],
            "assigned_agent": agent,
            "task_description": task_templates.get(agent, "Procesar issue"),
            "priority": triaged_issue['priority'],
            "complexity": triaged_issue['complexity']
        }
        
        print(f"\n📤 Tarea creada para {agent}")
        print(f"   {task['task_description']}")
        
        return task
    
    def prioritize_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Priorizar tareas basándose en urgencia y complejidad"""
        print(f"\n📊 Priorizando {len(tasks)} tareas...")
        
        # Orden de prioridad
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (priority_order.get(t['priority'], 99), -t['complexity'])
        )
        
        print("\n🎯 Orden de ejecución:")
        for i, task in enumerate(sorted_tasks, 1):
            print(f"   {i}. Issue #{task['issue_number']} - "
                  f"{task['priority']} - {task['assigned_agent']}")
        
        return sorted_tasks
    
    def generate_status_report(self, issues: List[Dict], tasks: List[Dict]) -> str:
        """Generar reporte de estado para el equipo"""
        report = "# GitHub Agent Team - Status Report\n\n"
        
        report += f"## 📊 Summary\n"
        report += f"- Total open issues analyzed: {len(issues)}\n"
        report += f"- Tasks created: {len(tasks)}\n\n"
        
        # Agrupar por agente
        by_agent = {}
        for task in tasks:
            agent = task['assigned_agent']
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(task)
        
        report += "## 👥 Task Distribution\n\n"
        for agent, agent_tasks in by_agent.items():
            report += f"### {agent.replace('_', ' ').title()}\n"
            report += f"- Assigned tasks: {len(agent_tasks)}\n"
            for task in agent_tasks:
                report += f"  - Issue #{task['issue_number']} ({task['priority']})\n"
            report += "\n"
        
        return report


def main():
    """Demostración del GitHub Agent Manager"""
    print("=" * 70)
    print("GitHub Agent Manager - Capstone Project")
    print("=" * 70)
    
    # Verificar configuración
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("\n⚠️ GITHUB_TOKEN no configurado.")
        print("Este es un ejemplo educativo. En producción necesitarías un token real.")
        print("Usando modo DEMO con datos simulados...\n")
        
        # Datos simulados
        simulated_issues = [
            {"number": 123, "title": "Login button not working", "body": "Users can't login", "labels": ["bug"], "created_at": "2024-01-01"},
            {"number": 124, "title": "Add dark mode support", "body": "Feature request for dark theme", "labels": ["enhancement"], "created_at": "2024-01-02"},
            {"number": 125, "title": "Update API documentation", "body": "Docs are outdated", "labels": ["documentation"], "created_at": "2024-01-03"},
        ]
        
        # Simular workflow
        print("\n📋 Simulando issues de GitHub...")
        manager = type('obj', (object,), {'llm': LLM})()  # Mock manager
        
        triaged_issues = []
        for issue in simulated_issues:
            print(f"\n   Issue #{issue['number']}: {issue['title']}")
            triaged = {
                **issue,
                "type": "bug" if "bug" in issue['labels'] else "enhancement",
                "priority": "high",
                "assign_to": "code_analyst" if "bug" in issue['labels'] else "code_writer",
                "complexity": 5,
                "summary": issue['title']
            }
            triaged_issues.append(triaged)
        
        print("\n✅ Demo workflow completado")
        print(f"\n💡 En producción, este manager:")
        print("   1. Obtendría issues reales de GitHub")
        print("   2. Las clasificaría con IA")
        print("   3. Las asignaría a agentes especializados")
        print("   4. Orquestaría el flujo completo de resolución")
        
        return
    
    # Workflow real (si hay token)
    repo_name = os.getenv("GITHUB_REPO", "owner/repo")
    manager = GitHubAgentManager(github_token, repo_name)
    
    # Obtener issues
    issues = manager.fetch_open_issues(limit=5)
    
    # Triage
    triaged_issues = []
    for issue in issues:
        triaged = manager.triage_issue(issue)
        triaged_issues.append(triaged)
    
    # Crear tareas
    tasks = [manager.create_task_for_agent(issue) for issue in triaged_issues]
    
    # Priorizar
    prioritized_tasks = manager.prioritize_tasks(tasks)
    
    # Generar reporte
    report = manager.generate_status_report(issues, tasks)
    
    print(f"\n{'=' * 70}")
    print("STATUS REPORT")
    print('=' * 70)
    print(report)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY no configurada")
    
    main()
