"""
01_mcp_server_simple.py
=======================
Servidor MCP Básico.

Este script implementa un servidor compatible con el Model Context Protocol.
Expone dos herramientas para gestionar un inventario simulado.

Requisitos:
pip install mcp
"""

from typing import Any
import asyncio
from mcp.server.fastmcp import FastMCP

# Inicializar el servidor FastMCP
mcp = FastMCP("InventarioServer")

# Datos simulados (Base de Datos en memoria)
INVENTORY = {
    "laptop_pro": {"price": 1200, "stock": 50},
    "mouse_gamer": {"price": 50, "stock": 100},
    "monitor_4k": {"price": 400, "stock": 20}
}

@mcp.tool()
def list_products() -> str:
    """Lista todos los productos disponibles en el inventario."""
    return "\n".join([f"- {k}: ${v['price']} (Stock: {v['stock']})" for k, v in INVENTORY.items()])

@mcp.tool()
def get_product_details(product_name: str) -> str:
    """Obtiene detalles específicos de un producto por su nombre."""
    product = INVENTORY.get(product_name)
    if product:
        return f"Detalles de {product_name}:\nPrecio: ${product['price']}\nStock: {product['stock']}"
    return f"Error: Producto '{product_name}' no encontrado."

@mcp.tool()
def update_stock(product_name: str, quantity: int) -> str:
    """Actualiza el stock de un producto (puede ser negativo para restar)."""
    if product_name not in INVENTORY:
        return f"Error: Producto '{product_name}' no existe."
    
    current_stock = INVENTORY[product_name]["stock"]
    new_stock = current_stock + quantity
    
    if new_stock < 0:
        return f"Error: No hay suficiente stock. Stock actual: {current_stock}"
    
    INVENTORY[product_name]["stock"] = new_stock
    return f"Stock actualizado. {product_name} ahora tiene {new_stock} unidades."

if __name__ == "__main__":
    print("🚀 Iniciando Servidor MCP de Inventario...")
    # En un entorno real, esto se ejecutaría sobre stdio o SSE
    mcp.run()
