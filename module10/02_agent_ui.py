"""
02_agent_ui.py
==============
Interfaz Profesional en Streamlit.

Características Avanzadas:
1.  **Configuración Dinámica:** Sidebar para ajustar parámetros del modelo.
2.  **Manejo de Errores Robusto:** Feedback visual si la API falla.
3.  **Estado Persistente:** Mantiene la conversación entre recargas.
4.  **Diseño Limpio:** Uso de columnas y contenedores para layout.

Requisitos:
pip install streamlit requests
"""

import streamlit as st
import requests
import json
import time

# --- Configuración ---
API_URL = "http://localhost:8000/chat/stream"
PAGE_TITLE = "Enterprise Agent Portal"
PAGE_ICON = "🏢"

st.set_page_config(
    page_title=PAGE_TITLE, 
    page_icon=PAGE_ICON, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Configuración ---
with st.sidebar:
    st.title(f"{PAGE_ICON} Configuración")
    st.markdown("---")
    
    model_temp = st.slider(
        "Creatividad (Temperatura)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Valores altos generan respuestas más variadas."
    )
    
    user_id = st.text_input("ID de Usuario", value="admin_user")
    
    st.markdown("---")
    if st.button("🗑️ Limpiar Historial", type="primary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("### 📊 Estado del Sistema")
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            st.success("API Online 🟢")
        else:
            st.warning("API Inestable 🟡")
    except:
        st.error("API Offline 🔴")

# --- Main: Chat ---
st.title(PAGE_TITLE)
st.caption("Interfaz conectada a Backend FastAPI Asíncrono con Streaming SSE.")

# Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderizar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    # 1. Guardar y mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Llamada a la API y Streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Payload con configuración
        payload = {
            "query": prompt,
            "user_id": user_id,
            "temperature": model_temp
        }
        
        start_time = time.time()
        try:
            with requests.post(API_URL, json=payload, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    st.error(f"Error del servidor: {r.status_code}")
                else:
                    # Procesar SSE (Server-Sent Events)
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                token = decoded_line[6:]
                                if token == "[DONE]":
                                    break
                                if token.startswith("[ERROR]"):
                                    st.error(token)
                                    break
                                    
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")
                                
                    response_placeholder.markdown(full_response)
                    
                    # Guardar respuesta
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Métricas
                    latency = time.time() - start_time
                    st.caption(f"⏱️ Respuesta generada en {latency:.2f}s")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar al Backend. Asegúrate de que `01_agent_api.py` esté corriendo.")
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
