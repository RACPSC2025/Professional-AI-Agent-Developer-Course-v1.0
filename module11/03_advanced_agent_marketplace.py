"""
Módulo 11 - Ejemplo Avanzado: Marketplace de Agentes (Agent-to-Agent)
Caso de uso: Marketplace donde agentes compran/venden servicios y negocian

Este ejemplo demuestra comunicación Agent-to-Agent (A2A) donde agentes
autónomos pueden descubrir servicios, negociar precios y realizar transacciones.

Instalación:
pip install langchain langchain-openai
"""

import os
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class MessageType(Enum):
    """Tipos de mensajes A2A (basado en FIPA)"""
    REQUEST = "request"  # Solicitar servicio
    PROPOSE = "propose"  # Proponer términos
    ACCEPT = "accept"  # Aceptar propuesta
    REJECT = "reject"  # Rechazar
    INFORM = "inform"  # Informar resultados


@dataclass
class A2AMessage:
    """Mensaje Agent-to-Agent estandarizado"""
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict
    conversation_id: str


class MarketplaceAgent:
    """Agente que participa en el marketplace"""
    
    def __init__(self, name: str, role: str, capabilities: List[str], budget: float = 0):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.budget = budget
        self.inbox = []
        self.conversation_history = {}
        
        print(f"✅ Agente '{name}' creado ({role})")
        print(f"   Capacidades: {', '.join(capabilities)}")
        if budget > 0:
            print(f"   Presupuesto: ${budget}")
    
    async def send_message(self, recipient: 'MarketplaceAgent', msg: A2AMessage):
        """Enviar mensaje a otro agente"""
        recipient.inbox.append(msg)
        
        # Registrar en historial
        if msg.conversation_id not in self.conversation_history:
            self.conversation_history[msg.conversation_id] = []
        self.conversation_history[msg.conversation_id].append(msg)
        
        print(f"\n📤 {self.name} → {recipient.name}: {msg.message_type.value.upper()}")
        print(f"   {msg.content}")
    
    async def process_inbox(self):
        """Procesar mensajes recibidos"""
        while self.inbox:
            msg = self.inbox.pop(0)
            await self.handle_message(msg)
    
    async def handle_message(self, msg: A2AMessage):
        """Manejar mensaje recibido (a ser implementado por subclases)"""
        print(f"\n📥 {self.name} recibió: {msg.message_type.value} de {msg.sender}")


class ServiceProvider(MarketplaceAgent):
    """Agente que ofrece servicios"""
    
    def __init__(self, name: str, services: Dict[str, float]):
        super().__init__(name, "Service Provider", list(services.keys()))
        self.services = services  # {service_name: base_price}
    
    async def handle_message(self, msg: A2AMessage):
        """Procesar solicitudes de servicio"""
        await super().handle_message(msg)
        
        if msg.message_type == MessageType.REQUEST:
            # Extraer servicio solicitado
            requested_service = msg.content.get("service")
            max_budget = msg.content.get("budget", 999999)
            
            if requested_service in self.services:
                base_price = self.services[requested_service]
                
                # Negociar precio (puede ajustar basado en demanda, cliente, etc.)
                final_price = self.negotiate_price(base_price, max_budget)
                
                if final_price <= max_budget:
                    # Enviar propuesta
                    response = A2AMessage(
                        sender=self.name,
                        receiver=msg.sender,
                        message_type=MessageType.PROPOSE,
                        content={
                            "service": requested_service,
                            "price": final_price,
                            "delivery_time_hours": 24
                        },
                       conversation_id=msg.conversation_id
                    )
                else:
                    # Rechazar
                    response = A2AMessage(
                        sender=self.name,
                        receiver=msg.sender,
                        message_type=MessageType.REJECT,
                        content={"reason": "Price exceeds budget"},
                        conversation_id=msg.conversation_id
                    )
                
                # Enviar respuesta (simulado - en realidad buscaríamos el agente)
                print(f"   💰 Proponiendo ${final_price} por {requested_service}")
        
        elif msg.message_type == MessageType.ACCEPT:
            # Servicio aceptado - ejecutar
            print(f"   ✅ {self.name}: Iniciando ejecución del servicio...")
    
    def negotiate_price(self, base_price: float, max_budget: float) -> float:
        """Negociar precio basado en presupuesto del cliente"""
        # Estrategia simple: ofrecer 90% del precio base o max budget
        return min(base_price * 0.9, max_budget)


class ServiceBuyer(MarketplaceAgent):
    """Agente que busca y compra servicios"""
    
    def __init__(self, name: str, budget: float):
        super().__init__(name, "Service Buyer", ["procurement"], budget)
        self.best_offer = None
    
    async def request_service(self, service_name: str, providers: List[ServiceProvider]):
        """Solicitar servicio a múltiples proveedores"""
        conversation_id = f"conv_{self.name}_{service_name}"
        
        print(f"\n🔍 {self.name} buscando: {service_name} (presupuesto: ${self.budget})")
        
        # Enviar REQUEST a todos los proveedores
        for provider in providers:
            msg = A2AMessage(
                sender=self.name,
                receiver=provider.name,
                message_type=MessageType.REQUEST,
                content={
                    "service": service_name,
                    "budget": self.budget
                },
                conversation_id=conversation_id
            )
            
            await self.send_message(provider, msg)
            await provider.process_inbox()
        
        # Simular recepción de propuestas
        print(f"\n💼 {self.name} evaluando propuestas...")
    
    async def handle_message(self, msg: A2AMessage):
        """Procesar propuestas de proveedores"""
        await super().handle_message(msg)
        
        if msg.message_type == MessageType.PROPOSE:
            offer = msg.content
            
            # Evaluar si es mejor oferta
            if self.best_offer is None or offer["price"] < self.best_offer["price"]:
                self.best_offer = {**offer, "provider": msg.sender}
                print(f"   ⭐ Nueva mejor oferta: ${offer['price']} de {msg.sender}")


async def main():
    """Demostración de marketplace de agentes"""
    print("=" * 70)
    print("Marketplace de Agentes - Comunicación Agent-to-Agent (A2A)")
    print("=" * 70)
    
    # Crear proveedores de servicios
    provider1 = ServiceProvider(
        name="DataAnalytics_Co",
        services={
            "data_analysis": 500.0,
            "visualization": 300.0,
            "reporting": 200.0
        }
    )
    
    provider2 = ServiceProvider(
        name="ML_Solutions",
        services={
            "data_analysis": 450.0,
            "model_training": 800.0,
            "deployment": 600.0
        }
    )
    
    provider3 = ServiceProvider(
        name="QuickAnalytics",
        services={
            "data_analysis": 350.0,
            "basic_stats": 150.0
        }
    )
    
    # Crear comprador
    buyer = ServiceBuyer(name="Enterprise_Client", budget=400.0)
    
    # Escenario: Buyer busca servicio de análisis de datos
    print(f"\n{'=' * 70}")
    print("ESCENARIO: Búsqueda y Negociación de Servicio")
    print('=' * 70)
    
    await buyer.request_service("data_analysis", [provider1, provider2, provider3])
    
    # Procesar respuestas
    await buyer.process_inbox()
    
    # Decisión final
    print(f"\n{'=' * 70}")
    print("DECISIÓN FINAL")
    print('=' * 70)
    
    if buyer.best_offer:
        print(f"\n🏆 {buyer.name} selecciona:")
        print(f"   Proveedor: {buyer.best_offer['provider']}")
        print(f"   Servicio: {buyer.best_offer['service']}")
        print(f"   Precio: ${buyer.best_offer['price']}")
        print(f"   Tiempo de entrega: {buyer.best_offer['delivery_time_hours']}h")
        
        # Enviar aceptación
        print(f"\n✅ Transacción completada mediante A2A protocol")
    
    print(f"\n{'=' * 70}")
    print("Demostración completada")
    print('=' * 70)
    
    print("""
💡 CARACTERÍSTICAS DEL PROTOCOLO A2A:

✅ Comunicación estandarizada entre agentes autónomos
✅ Negociación automática de términos y precios
✅ Descubrimiento de servicios en el marketplace
✅ Conversaciones multipartitas rastreables
✅ Extensible a blockchains para transacciones reales

🌐 CASOS DE USO REALES:
   - Marketplaces de servicios de IA
   - Trading automatizado entre agentes
   - Coordinación de fleets de robots/drones
   - Supply chain automation
   - Subastas y procurement automático
""")


if __name__ == "__main__":
    asyncio.run(main())
