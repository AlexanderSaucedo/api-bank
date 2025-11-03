import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine
import os

# === Configuración ===
st.set_page_config(page_title="Dashboard SVM", layout="wide")
st.title("📊 Dashboard - Bank Marketing (SVM)")

API_URL = os.getenv("API_URL", "web-production-271f3.up.railway.app")
DB_URL = os.getenv("DATABASE_URL")

# === Cargar métricas desde API ===
st.subheader("📈 Métricas del Modelo")
try:
    r = requests.get(f"{API_URL}/metrics")
    data = r.json()
    st.json(data)
except Exception as e:
    st.error(f"No se pudo conectar con la API: {e}")

# === Conectar a PostgreSQL ===
try:
    if DB_URL.startswith("postgres://"):
        DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM resultados_svm", engine)
    st.subheader("📊 Datos guardados en la Base de Datos")
    st.dataframe(df)
except Exception as e:
    st.warning(f"No se pudo conectar a la base de datos: {e}")
