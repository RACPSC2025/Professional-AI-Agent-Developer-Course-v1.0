"""
🟡 NIVEL INTERMEDIO: AGENTE CON ACCESO A BASE DE DATOS
------------------------------------------------------
Este ejemplo demuestra cómo crear un agente que puede consultar y modificar una base de datos SQLite.
Caso de Uso: Sistema CRM simple que gestiona clientes y sus pedidos.

Conceptos Clave:
- SQL Injection Prevention (prepared statements).
- Transacciones seguras para operaciones de escritura.
- Human-in-the-Loop para operaciones críticas.
"""

import os
import sys
import sqlite3
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. INICIALIZACIÓN DE BASE DE DATOS ---
DB_PATH = "crm_database.db"

def init_database():
    """Crea la base de datos de ejemplo si no existe."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabla de clientes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tabla de pedidos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            total_price REAL NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)
    
    # Datos de ejemplo
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        sample_customers = [
            ("Juan Pérez", "juan.perez@email.com", "+34600123456"),
            ("María García", "maria.garcia@email.com", "+34600654321"),
            ("Carlos Rodríguez", "carlos.rodriguez@email.com", "+34600987654")
        ]
        cursor.executemany("INSERT INTO customers (name, email, phone) VALUES (?, ?, ?)", sample_customers)
        
        sample_orders = [
            (1, "Laptop Pro", 1, 1200.00),
            (1, "Mouse Inalámbrico", 2, 30.00),
            (2, "Monitor 27 pulgadas", 1, 350.00)
        ]
        cursor.executemany("INSERT INTO orders (customer_id, product, quantity, total_price) VALUES (?, ?, ?, ?)", sample_orders)
    
    conn.commit()
    conn.close()
    print("✅ Base de datos inicializada.")

init_database()

# --- 2. DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---

class SearchCustomerInput(BaseModel):
    """Input para buscar clientes."""
    query: str = Field(description="Nombre o email del cliente a buscar")

class CreateOrderInput(BaseModel):
    """Input para crear un pedido."""
    customer_email: str = Field(description="Email del cliente")
    product: str = Field(description="Nombre del producto")
    quantity: int = Field(description="Cantidad de unidades")
    price: float = Field(description="Precio total del pedido")

@tool("search_customers", args_schema=SearchCustomerInput)
def search_customers(query: str) -> str:
    """
    Busca clientes por nombre o email en la base de datos.
    Usa prepared statements para prevenir SQL injection.
    """
    print(f"🔍 Buscando clientes con: '{query}'...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Prepared statement (seguro contra SQL injection)
        cursor.execute("""
            SELECT id, name, email, phone
            FROM customers
            WHERE name LIKE ? OR email LIKE ?
        """, (f"%{query}%", f"%{query}%"))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No se encontraron clientes con '{query}'."
        
        output = f"Encontrados {len(results)} cliente(s):\n"
        for customer_id, name, email, phone in results:
            output += f"\n  ID: {customer_id}\n  Nombre: {name}\n  Email: {email}\n  Teléfono: {phone}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error al buscar clientes: {str(e)}"

@tool("get_customer_orders")
def get_customer_orders(customer_email: str) -> str:
    """
    Obtiene todos los pedidos de un cliente por su email.
    """
    print(f"📦 Consultando pedidos de {customer_email}...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT o.id, o.product, o.quantity, o.total_price, o.order_date
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE c.email = ?
        """, (customer_email,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No hay pedidos para {customer_email}."
        
        output = f"Pedidos de {customer_email}:\n"
        total = 0
        for order_id, product, quantity, price, order_date in results:
            output += f"\n  Pedido #{order_id}: {quantity}x {product} - €{price:.2f} ({order_date})"
            total += price
        
        output += f"\n\nTotal gastado: €{total:.2f}"
        return output
        
    except Exception as e:
        return f"❌ Error al consultar pedidos: {str(e)}"

@tool("create_order", args_schema=CreateOrderInput)
def create_order(customer_email: str, product: str, quantity: int, price: float) -> str:
    """
    Crea un nuevo pedido para un cliente.
    ⚠️ OPERACIÓN DE ESCRITURA: Requiere confirmación del usuario.
    """
    print(f"🛒 Creando pedido: {quantity}x {product} para {customer_email}...")
    
    # HUMAN-IN-THE-LOOP: Solicitar confirmación
    print(f"\n⚠️  CONFIRMACIÓN REQUERIDA:")
    print(f"   Cliente: {customer_email}")
    print(f"   Producto: {quantity}x {product}")
    print(f"   Total: €{price:.2f}")
    
    confirm = input("   ¿Confirmar creación? (s/n): ")
    
    if confirm.lower() != 's':
        return "❌ Creación de pedido cancelada por el usuario."
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Obtener ID del cliente
        cursor.execute("SELECT id FROM customers WHERE email = ?", (customer_email,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return f"❌ Cliente con email {customer_email} no encontrado."
        
        customer_id = result[0]
        
        # Insertar pedido
        cursor.execute("""
            INSERT INTO orders (customer_id, product, quantity, total_price)
            VALUES (?, ?, ?, ?)
        """, (customer_id, product, quantity, price))
        
        order_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return f"✅ Pedido #{order_id} creado exitosamente para {customer_email}."
        
    except Exception as e:
        return f"❌ Error al crear pedido: {str(e)}"

# --- 3. AGENTE ---
tools = [search_customers, get_customer_orders, create_order]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Eres un Asistente de CRM profesional. 🏢

CAPACIDADES:
- Buscar clientes en la base de datos.
- Consultar pedidos de un cliente.
- Crear nuevos pedidos (requiere confirmación).

REGLAS:
1. NUNCA crees pedidos sin verificar primero que el cliente existe.
2. SIEMPRE consulta la información antes de dar respuestas.
3. Para creaciones, sé claro sobre lo que se va a hacer.
    """),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. INTERFAZ ---
if __name__ == "__main__":
    print("--- 🏢 SISTEMA CRM PROFESIONAL ---")
    print("Base de datos: crm_database.db")
    print("\nEjemplos:")
    print("  - '¿Qué clientes tengo?'")
    print("  - 'Muéstrame los pedidos de juan.perez@email.com'")
    print("  - 'Crear un pedido de 2 teclados a €50 para María'\n")
    
    while True:
        query = input("\n💼 Consulta (o 'salir'): ")
        if query.lower() in ["salir", "exit"]:
            break
        
        try:
            response = agent_executor.invoke({"input": query})
            print(f"\n📋 RESPUESTA:\n{response['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
