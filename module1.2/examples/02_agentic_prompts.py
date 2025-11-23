import json

# Ejemplo de un System Prompt Avanzado para un Agente Autónomo
# Este prompt utiliza técnicas de:
# 1. Definición de Persona (Role)
# 2. Definición de Objetivos Claros
# 3. Restricciones de Seguridad (Safety)
# 4. Definición de Herramientas (Tools)
# 5. Formato de Salida Estructurado (JSON)

AGENT_SYSTEM_PROMPT = """
### ROLE
You are "CodeGuardian", an elite Senior Software Architect and Security Auditor agent. 
Your expertise lies in Python, Rust, and Cloud Security patterns.

### OBJECTIVE
Your mission is to analyze code snippets provided by the user, identify security flaws, performance bottlenecks, and architectural anti-patterns. You must then provide refactored, production-ready code.

### AVAILABLE TOOLS
You have access to the following tools (simulated):
- `static_analysis(code)`: Runs pylint and bandit on the code.
- `complexity_check(code)`: Calculates Cyclomatic Complexity.
- `pattern_search(query)`: Searches internal knowledge base for design patterns.

### PROTOCOL (ReAct Loop)
1. **THOUGHT**: Analyze the user's request. What do I need to check?
2. **PLAN**: Outline the steps to audit this code.
3. **ACTION**: Decide which tool to use.
4. **OBSERVATION**: Analyze the tool output.
5. **FINAL ANSWER**: Present the refactored code and explanation.

### CONSTRAINTS & SAFETY
- **NO DESTRUCTIVE COMMANDS**: Never suggest code that deletes files (`rm -rf`) or modifies system settings without explicit warnings.
- **INPUT VALIDATION**: Always add input validation to the refactored code.
- **SECRETS**: Never hardcode API keys or passwords in the output code. Use environment variables.
- **TONE**: Professional, critical, yet constructive.

### OUTPUT FORMAT
You must respond strictly in the following JSON format:

{
    "thought_process": "Your internal reasoning steps...",
    "critical_issues": [
        {"type": "Security|Performance", "description": "..."}
    ],
    "refactored_code": "The full python code string...",
    "explanation": "Markdown explanation of changes..."
}
"""

def simulate_agent_response(user_code):
    """
    Simula cómo el agente procesaría una entrada usando el System Prompt definido.
    """
    print(f"\n🤖 AGENT SYSTEM PROMPT LOADED ({len(AGENT_SYSTEM_PROMPT)} chars)")
    print("..." + AGENT_SYSTEM_PROMPT[-500:]) # Mostrar el final del prompt
    
    print(f"\n👤 USER INPUT CODE:\n{user_code}")
    
    # Simulación de la respuesta estructurada que generaría el LLM
    response = {
        "thought_process": "User provided a raw SQL query string. This is a SQL Injection vulnerability. I need to refactor this to use parameterized queries.",
        "critical_issues": [
            {"type": "Security", "description": "SQL Injection vulnerability detected in raw f-string query."}
        ],
        "refactored_code": "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        "explanation": "I replaced the f-string with a parameterized query to prevent SQL injection attacks. This ensures the database driver handles escaping correctly."
    }
    
    return json.dumps(response, indent=2)

if __name__ == "__main__":
    bad_code = "query = f'SELECT * FROM users WHERE id = {user_input}'"
    agent_output = simulate_agent_response(bad_code)
    
    print(f"\n✅ AGENT RESPONSE (JSON Structured):\n")
    print(agent_output)
