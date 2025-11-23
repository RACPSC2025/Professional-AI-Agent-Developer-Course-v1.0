import os
import time

# Simulación de una llamada a un LLM (para no requerir API Key real en este ejemplo básico)
# En un caso real, usaríamos openai.ChatCompletion.create() o similar.

def mock_llm_call(prompt, system_message=""):
    """
    Simula la respuesta de un LLM basada en la calidad del prompt.
    """
    full_input = f"System: {system_message}\nUser: {prompt}"
    
    print(f"\n{'='*50}")
    print(f"📥 ENVIANDO PROMPT AL MODELO...")
    print(f"{'='*50}")
    print(f"System: {system_message}")
    print(f"User: {prompt}")
    print(f"{'-'*50}")
    
    # Respuestas simuladas para demostrar la diferencia
    if "R-A-C-E" in system_message or "tabla" in prompt.lower():
        return """
✅ RESPUESTA (Estructurada y Profesional):

| Vulnerabilidad | Severidad | Solución Sugerida |
| :--- | :--- | :--- |
| Puerto 22 abierto a 0.0.0.0/0 | CRÍTICA | Restringir acceso SSH solo a la VPN corporativa o IPs específicas. |
| Falta de cifrado en tránsito (HTTP) | ALTA | Habilitar TLS/SSL y redirigir todo el tráfico a HTTPS. |
| Permisos de S3 bucket públicos | ALTA | Bloquear acceso público y usar políticas de bucket restrictivas. |
        """
    elif "R-I-S-E" in system_message:
        return """
✅ RESPUESTA (Chain-of-Thought):

1. **Identificación de Hechos**: El usuario pregunta sobre la legalidad de usar datos de scraping para entrenar IA.
2. **Búsqueda de Precedentes**: Analizando casos recientes como NYT vs OpenAI (2023).
3. **Análisis**: Existe una zona gris legal. El "Fair Use" es el argumento principal de defensa, pero no está garantizado.
4. **Conclusión**: Se recomienda precaución y uso de datasets con licencia explícita.

**MEMORÁNDUM LEGAL**
Para: Cliente
De: Asistente Legal AI
Asunto: Riesgos de Copyright en Entrenamiento de IA
...
        """
    else:
        return """
❌ RESPUESTA (Genérica y Pobre):
Hola. Pues mira, deberías revisar que no tengas puertos abiertos y esas cosas. El cifrado es importante también. Ten cuidado con los buckets de S3. Saludos.
        """

def main():
    print("🧪 LABORATORIO DE PROMPT ENGINEERING: Frameworks en Acción\n")

    # CASO 1: Prompt Básico (Sin Framework)
    print("\n🔴 CASO 1: Prompt Básico (Sin Estructura)")
    basic_system = "Eres un asistente útil."
    basic_prompt = "Dime qué está mal en este archivo de configuración de seguridad."
    response_basic = mock_llm_call(basic_prompt, basic_system)
    print(response_basic)
    time.sleep(1)

    # CASO 2: Framework R-A-C-E
    print("\n🟢 CASO 2: Framework R-A-C-E (Role, Action, Context, Expectation)")
    race_system = """
    (R-A-C-E Framework Applied)
    ROLE: Eres un Ingeniero de Ciberseguridad Senior especializado en Cloud Infrastructure.
    CONTEXT: Estás auditando una infraestructura crítica para un banco.
    """
    race_prompt = """
    ACTION: Analiza las vulnerabilidades comunes en configuraciones de nube.
    EXPECTATION: Genera una tabla Markdown con las 3 vulnerabilidades más comunes, su severidad y solución técnica.
    """
    response_race = mock_llm_call(race_prompt, race_system)
    print(response_race)
    time.sleep(1)

    # CASO 3: Framework R-I-S-E
    print("\n🔵 CASO 3: Framework R-I-S-E (Role, Input, Steps, Expectation)")
    rise_system = """
    (R-I-S-E Framework Applied)
    ROLE: Eres un Asistente Legal experto en Propiedad Intelectual y Tecnología.
    """
    rise_prompt = """
    INPUT: Pregunta sobre scraping para IA.
    STEPS: 1. Identifica hechos. 2. Busca precedentes. 3. Analiza. 4. Concluye.
    EXPECTATION: Un resumen estructurado paso a paso y un memorándum formal.
    """
    response_rise = mock_llm_call(rise_prompt, rise_system)
    print(response_rise)

if __name__ == "__main__":
    main()
