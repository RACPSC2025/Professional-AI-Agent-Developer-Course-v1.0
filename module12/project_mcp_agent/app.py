import streamlit as st
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

# Configuración de la página
st.set_page_config(page_title="Mi Agente MCP Personal", page_icon="🤖", layout="wide")

st.title("🤖 Mi Agente MCP Personal")
st.markdown("### Tu Centro de Comando Seguro")

# Sidebar: Configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    api_key = st.text_input("Tu API Key (OpenAI/Anthropic)", type="password")
    model = st.selectbox("Modelo LLM", ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"])
    
    st.divider()
    
    st.subheader("🔌 Estado MCP")
    if st.button("Conectar Servidor Local"):
        st.session_state.mcp_connected = True
        st.success("Servidor 'server.py' conectado!")

# Estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visualizador de Herramientas (Simulado para UI, real en backend)
if st.session_state.get("mcp_connected"):
    with st.expander("🛠️ Herramientas MCP Disponibles (Live)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1: st.info("📧 gmail_send")
        with col2: st.info("🔒 encrypt_data")
        with col3: st.info("🕒 get_server_time")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu orden (ej: 'Envía un correo encriptado...')"):
    # 1. Mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Simulación de Respuesta del Agente (Aquí iría la lógica real con LangChain + MCP)
    with st.chat_message("assistant"):
        if not api_key:
            st.error("⚠️ Por favor ingresa tu API Key para activar el cerebro del Agente.")
        elif not st.session_state.get("mcp_connected"):
            st.warning("⚠️ El servidor MCP no está conectado.")
        else:
            response_text = f"🤖 **Agente ({model}):** Procesando tu solicitud '{prompt}' usando las herramientas MCP..."
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Simulación de uso de herramienta
            if "correo" in prompt.lower():
                with st.status("📧 Usando herramienta: gmail_send..."):
                    st.write("Conectando con API de Gmail...")
                    st.write("Redactando mensaje...")
                    st.write("Enviando...")
                st.success("✅ Correo enviado exitosamente.")
