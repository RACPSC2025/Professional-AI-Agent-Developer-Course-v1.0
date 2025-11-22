"""
frontend/app.py
===============
Frontend "Software House" en Streamlit.
Interfaz moderna y oscura para interactuar con el equipo de agentes.
"""

import streamlit as st
import requests
import json

# --- Configuración de Página ---
st.set_page_config(
    page_title="Autonomous Software House",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Personalizado (Cyberpunk/Dark Theme) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .agent-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: #1E1E1E;
        border-left: 5px solid #8E44AD;
    }
    .success-card {
        border-left: 5px solid #2ECC71;
    }
    .error-card {
        border-left: 5px solid #E74C3C;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🏗️ Autonomous Software House")
st.markdown("### *Tú lo imaginas, nuestros agentes lo construyen.*")

# --- Input ---
col1, col2 = st.columns([3, 1])
with col1:
    requirement = st.text_area(
        "Describe el software que necesitas:",
        placeholder="Ej: Un script de Python que analice precios de Bitcoin y me envíe una alerta si baja de $90k.",
        height=100
    )
with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    start_btn = st.button("🚀 Iniciar Proyecto", type="primary", use_container_width=True)

# --- Lógica de Ejecución ---
if start_btn and requirement:
    st.divider()
    st.subheader("👷 Progreso del Equipo")
    
    logs_container = st.container()
    code_container = st.empty()
    
    API_URL = "http://localhost:8000/create-software"
    
    try:
        with requests.post(API_URL, json={"requirement": requirement}, stream=True) as r:
            if r.status_code == 200:
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            msg = decoded_line[6:]
                            
                            if msg == "[DONE]":
                                st.success("✨ Proyecto Completado Exitosamente.")
                                break
                            
                            if msg.startswith("CODE_BLOCK: "):
                                # Extraer y mostrar código final
                                code = msg.replace("CODE_BLOCK: ", "").replace("\\n", "\n")
                                code_container.code(code, language="python")
                                continue
                                
                            # Mostrar logs de agentes con estilo
                            with logs_container:
                                if "PM" in msg:
                                    st.markdown(f"<div class='agent-card'>👔 {msg}</div>", unsafe_allow_html=True)
                                elif "Coder" in msg:
                                    st.markdown(f"<div class='agent-card' style='border-color: #3498DB;'>👨‍💻 {msg}</div>", unsafe_allow_html=True)
                                elif "QA" in msg:
                                    color = "#2ECC71" if "✅" in msg else "#E74C3C"
                                    st.markdown(f"<div class='agent-card' style='border-color: {color};'>🧐 {msg}</div>", unsafe_allow_html=True)
                                else:
                                    st.info(msg)
                                    
            else:
                st.error(f"Error del servidor: {r.status_code}")
    except Exception as e:
        st.error(f"No se pudo conectar a la Software House. Asegúrate de correr el backend: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Powered by LangGraph, FastAPI & CrewAI | Module 13 Capstone")
