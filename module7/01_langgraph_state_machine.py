"""
🟢 NIVEL BÁSICO: LANGGRAPH - MÁQUINA DE ESTADOS PARA AGENTES
------------------------------------------------------------
Este ejemplo demuestra LangGraph para crear agentes con flujos complejos y condicionales.
Caso de Uso: Agente de procesamiento de pedidos con múltiples estados y validaciones.

Conceptos Clave:
- State Graph: Grafo de estados dirigido acíclico (DAG)
- Conditional edges: Transiciones basadas en condiciones
- State management: Gestión de estado compartido entre nodos
"""

import os
import sys
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY no configurada.")
    sys.exit(1)

# --- 1. DEFINIR ESTADO (STATE) ---
class OrderState(TypedDict):
    """Estado compartido del agente de pedidos."""
    messages: Annotated[list, add_messages]  # Historial de mensajes
    customer_verified: bool
    inventory_checked: bool
    payment_processed: bool
    order_id: str
    status: str  # "pending", "processing", "completed", "failed"

# --- 2. DEFINIR NODOS (NODES) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def verify_customer(state: OrderState) -> OrderState:
    """Nodo 1: Verificar identidad del cliente."""
    print("🔍 NODO: Verificando cliente...")
    
    # Simulamos verificación (en producción, consultaría una BD)
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Extraer info del cliente del mensaje
    if "cliente" in last_message.lower() or "juan" in last_message.lower():
        state["customer_verified"] = True
        state["status"] = "customer_verified"
        state["messages"].append(AIMessage(content="✅ Cliente verificado."))
    else:
        state["customer_verified"] = False
        state["status"] = "verification_failed"
        state["messages"].append(AIMessage(content="❌ No se pudo verificar el cliente."))
    
    return state

def check_inventory(state: OrderState) -> OrderState:
    """Nodo 2: Verificar disponibilidad en inventario."""
    print("📦 NODO: Verificando inventario...")
    
    # Simulación
    state["inventory_checked"] = True
    state["status"] = "inventory_available"
    state["messages"].append(AIMessage(content="✅ Producto disponible en inventario."))
    
    return state

def process_payment(state: OrderState) -> OrderState:
    """Nodo 3: Procesar pago."""
    print("💳 NODO: Procesando pago...")
    
    # Simulación de pago
    import random
    payment_success = random.choice([True, True, True, False])  # 75% éxito
    
    if payment_success:
        state["payment_processed"] = True
        state["order_id"] = f"ORD-{random.randint(10000, 99999)}"
        state["status"] = "payment_success"
        state["messages"].append(AIMessage(content=f"✅ Pago procesado. Order ID: {state['order_id']}"))
    else:
        state["payment_processed"] = False
        state["status"] = "payment_failed"
        state["messages"].append(AIMessage(content="❌ Pago rechazado. Verifique su método de pago."))
    
    return state

def complete_order(state: OrderState) -> OrderState:
    """Nodo 4: Completar pedido."""
    print("🎉 NODO: Completando pedido...")
    
    state["status"] = "completed"
    state["messages"].append(AIMessage(content=f"🎉 ¡Pedido {state['order_id']} completado con éxito!"))
    
    return state

def handle_failure(state: OrderState) -> OrderState:
    """Nodo 5: Manejar falla."""
    print("⚠️ NODO: Manejando falla...")
    
    state["status"] = "failed"
    state["messages"].append(AIMessage(content="❌ Pedido fallido. Contacte soporte."))
    
    return state

# --- 3. DEFINIR TRANSICIONES CONDICIONALES ---

def should_continue_after_verification(state: OrderState) -> Literal["check_inventory", "handle_failure"]:
    """Decide si continuar después de verificación."""
    if state["customer_verified"]:
        return "check_inventory"
    else:
        return "handle_failure"

def should_continue_after_payment(state: OrderState) -> Literal["complete_order", "handle_failure"]:
    """Decide si continuar después del pago."""
    if state["payment_processed"]:
        return "complete_order"
    else:
        return "handle_failure"

# --- 4. CONSTRUIR EL GRAFO ---
workflow = StateGraph(OrderState)

# Agregar nodos
workflow.add_node("verify_customer", verify_customer)
workflow.add_node("check_inventory", check_inventory)
workflow.add_node("process_payment", process_payment)
workflow.add_node("complete_order", complete_order)
workflow.add_node("handle_failure", handle_failure)

# Definir punto de entrada
workflow.set_entry_point("verify_customer")

# Agregar aristas condicionales
workflow.add_conditional_edges(
    "verify_customer",
    should_continue_after_verification,
    {
        "check_inventory": "check_inventory",
        "handle_failure": "handle_failure"
    }
)

# Aristas normales
workflow.add_edge("check_inventory", "process_payment")

# Arista condicional después del pago
workflow.add_conditional_edges(
    "process_payment",
    should_continue_after_payment,
    {
        "complete_order": "complete_order",
        "handle_failure": "handle_failure"
    }
)

# Finalizar
workflow.add_edge("complete_order", END)
workflow.add_edge("handle_failure", END)

# Compilar el grafo
app = workflow.compile()

# --- 5. VISUALIZACIÓN DEL GRAFO (OPCIONAL) ---
# Para visualizar: app.get_graph().draw_mermaid_png() requiere graphviz
print("\n📊 ESTRUCTURA DEL GRAFO:")
print("""
[START]
   ↓
[verify_customer]
   ↓
   ├→ customer_verified? → [check_inventory] → [process_payment]
   │                                                ↓
   │                                                ├→ payment_success? → [complete_order] → [END]
   │                                                └→ payment_failed → [handle_failure] → [END]
   └→ verification_failed → [handle_failure] → [END]
""")

# --- 6. EJECUCIÓN ---
if __name__ == "__main__":
    print("="*70)
    print("  🛒 SISTEMA DE PROCESAMIENTO DE PEDIDOS - LANGGRAPH")
    print("="*70)
    
    # Estado inicial
    initial_state = {
        "messages": [HumanMessage(content="Quiero hacer un pedido como cliente Juan Pérez")],
        "customer_verified": False,
        "inventory_checked": False,
        "payment_processed": False,
        "order_id": "",
        "status": "pending"
    }
    
    print("\n🚀 Iniciando flujo de pedido...\n")
    
    # Ejecutar el grafo
    result = app.invoke(initial_state)
    
    # Mostrar resultado final
    print("\n" + "="*70)
    print("  📋 RESULTADO FINAL")
    print("="*70)
    print(f"Estado final: {result['status']}")
    print(f"Order ID: {result.get('order_id', 'N/A')}")
    print(f"\n💬 HISTORIAL DE MENSAJES:")
    for msg in result["messages"]:
        role = "👤 Usuario" if isinstance(msg, HumanMessage) else "🤖 Sistema"
        print(f"{role}: {msg.content}")
    
    print("\n" + "="*70)
    print("\n💡 VENTAJAS DE LANGGRAPH:")
    print("  1. Flujos complejos con lógica condicional")
    print("  2. Estado compartido entre nodos")
    print("  3. Trazabilidad y debugging fácil")
    print("  4. Ideal para procesos empresariales (workflows)")
