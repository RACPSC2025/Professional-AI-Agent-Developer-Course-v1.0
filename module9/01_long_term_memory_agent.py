"""
🟢 NIVEL BÁSICO: AGENTE CON MEMORIA A LARGO PLAZO
-------------------------------------------------
Este ejemplo implementa un agente que recuerda conversaciones previas y personaliza respuestas.
Caso de Uso: Asistente personal que aprende preferencias del usuario.

Conceptos Clave:
- Long-term memory: Memoria persistente entre sesiones
- Contexto acumulativo: Aprendizaje de preferencias
- Storage: SQLite para persistencia simple
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. SISTEMA DE MEMORIA ---
class LongTermMemory:
    """Gestiona memoria persistente del agente."""
    
    def __init__(self, db_path="agent_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializar base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de conversaciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                user_message TEXT,
                agent_response TEXT
            )
        """)
        
        # Tabla de preferencias (metadata sobre el usuario)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                preferences_json TEXT,
                last_updated TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, user_id: str, user_message: str, agent_response: str):
        """Guardar interacción."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (user_id, timestamp, user_message, agent_response)
            VALUES (?, ?, ?, ?)
        """, (user_id, datetime.now().isoformat(), user_message, agent_response))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, user_id: str, limit: int = 5):
        """Recuperar últimas conversaciones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_message, agent_response, timestamp
            FROM conversations
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"user": r[0], "agent": r[1], "time": r[2]} for r in reversed(results)]
    
    def update_preferences(self, user_id: str, preferences: dict):
        """Actualizar preferencias del usuario."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO preferences (user_id, preferences_json, last_updated)
            VALUES (?, ?, ?)
        """, (user_id, json.dumps(preferences, ensure_ascii=False), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_preferences(self, user_id: str) -> dict:
        """Obtener preferencias del usuario."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT preferences_json FROM preferences WHERE user_id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}

# --- 2. AGENTE CON MEMORIA ---
class MemoryAgent:
    """Agente que recuerda y aprende."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.memory = LongTermMemory()
        
        self.prompt = ChatPromptTemplate.from_template("""
Eres un asistente personal inteligente con memoria a largo plazo.

HISTORIAL DE CONVERSACIONES PREVIAS:
{history}

PREFERENCIAS CONOCIDAS DEL USUARIO:
{preferences}

INSTRUCCIONES:
1. Usa el historial para mantener contexto
2. Personaliza respuestas según preferencias
3. Si detectas nueva información sobre el usuario (nombre, gustos, etc.), anótalo
4. Sé proactivo: sugiere cosas basándote en lo que sabes del usuario

MENSAJE ACTUAL: {message}

RESPUESTA:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def invoke(self, user_id: str, message: str) -> str:
        """Procesar mensaje con contexto de memoria."""
        
        # Recuperar contexto
        history = self.memory.get_conversation_history(user_id)
        preferences = self.memory.get_preferences(user_id)
        
        # Formatear historial
        history_text = ""
        if history:
            history_text = "\n".join([
                f"[{h['time'][:16]}] Usuario: {h['user']}\n  Tú: {h['agent']}"
                for h in history
            ])
        else:
            history_text = "No hay conversaciones previas."
        
        # Formatear preferencias
        pref_text = json.dumps(preferences, indent=2, ensure_ascii=False) if preferences else "Ninguna información registrada aún."
        
        # Generar respuesta
        response = self.chain.invoke({
            "history": history_text,
            "preferences": pref_text,
            "message": message
        })
        
        # Guardar en memoria
        self.memory.save_conversation(user_id, message, response)
        
        # Actualizar preferencias si se detecta nueva info (muy simplificado)
        self._update_preferences_if_needed(user_id, message, preferences)
        
        return response
    
    def _update_preferences_if_needed(self, user_id: str, message: str, current_prefs: dict):
        """Detectar y actualizar preferencias automáticamente."""
        # Ejemplos muy básicos (en producción, usar NER o prompts específicos)
        
        if "me llamo" in message.lower() or "mi nombre es" in message.lower():
            # Extraer nombre (simplificado)
            words = message.split()
            if "llamo" in message.lower():
                idx = words.index([w for w in words if "llamo" in w.lower()][0])
                name = words[idx + 1] if idx + 1 < len(words) else None
                if name and name not in current_prefs.get("name", ""):
                    current_prefs["name"] = name.capitalize()
                    self.memory.update_preferences(user_id, current_prefs)
        
        if "me gusta" in message.lower() or "prefiero" in message.lower():
            # Agregar a lista de gustos
            if "likes" not in current_prefs:
                current_prefs["likes"] = []
            # Simplificado: agregamos el mensaje completo (en producción, extraer entidades)
            current_prefs["likes"].append(message)
            self.memory.update_preferences(user_id, current_prefs)

# --- 3. INTERFAZ ---
if __name__ == "__main__":
    print("="*70)
    print("  🧠 AGENTE CON MEMORIA A LARGO PLAZO")
    print("="*70)
    print("\n💡 Este agente recuerda conversaciones previas y aprende sobre ti.\n")
    
    agent = MemoryAgent()
    user_id = "user_123"  # En producción, autenticar usuarios
    
    print("🎮 Comandos especiales:")
    print("  'historial' - Ver conversaciones previas")
    print("  'preferencias' - Ver lo que el agente sabe de ti")
    print("  'reset' - Borrar memoria")
    print("  'salir' - Terminar\n")
    
    while True:
        message = input("💬 Tú: ")
        
        if message.lower() == "salir":
            break
        
        elif message.lower() == "historial":
            history = agent.memory.get_conversation_history(user_id, limit=10)
            print("\n📜 HISTORIAL:")
            for h in history:
                print(f"  [{h['time'][:16]}]")
                print(f"    Tú: {h['user']}")
                print(f"    Agente: {h['agent']}\n")
            continue
        
        elif message.lower() == "preferencias":
            prefs = agent.memory.get_preferences(user_id)
            print(f"\n🔖 PREFERENCIAS:")
            print(json.dumps(prefs, indent=2, ensure_ascii=False))
            print()
            continue
        
        elif message.lower() == "reset":
            # Borrar todo (cuidado en producción)
            import os
            if os.path.exists("agent_memory.db"):
                os.remove("agent_memory.db")
                agent.memory._init_db()
                print("🗑️ Memoria borrada.\n")
            continue
        
        # Procesar mensaje normal
        response = agent.invoke(user_id, message)
        print(f"🤖 Agente: {response}\n")
    
    print("\n👋 ¡Hasta luego! Tu memoria se ha guardado para la próxima vez.")
