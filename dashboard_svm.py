import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="Dashboard SVM",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard - Modelo SVM (Bank Marketing)")

# =========================
# SWITCH DE TEMA
# =========================
theme_mode = st.sidebar.radio("Selecciona el tema", ["Claro", "Oscuro"])

if theme_mode == "Oscuro":
    BG_COLOR = "#121212"
    TEXT_COLOR = "#FFFFFF"
    PRIMARY_COLOR = "#00BFFF"
    CARD_COLOR = "#1E1E1E"
    HEATMAP_CMAP = "cool"
else:
    BG_COLOR = "#FFFFFF"
    TEXT_COLOR = "#000000"
    PRIMARY_COLOR = "#007bff"
    CARD_COLOR = "#FFFFFF"
    HEATMAP_CMAP = "Blues"

# =========================
# ESTILOS CSS PERSONALIZADOS
# =========================
st.markdown(f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

[data-testid="stAppViewContainer"], 
[data-testid="stMainContainer"], 
[data-testid="stHeader"], 
[data-testid="stSidebar"] {{
    background-color: {BG_COLOR} !important;
    color: {TEXT_COLOR} !important;
}}

h1, h2, h3 {{
    color: {PRIMARY_COLOR};
    font-weight: 600;
}}

.metric-card {{
    background-color: {CARD_COLOR};
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid {PRIMARY_COLOR};
    margin-bottom: 20px;
}}

.metric-value {{
    font-size: 2.5em;
    font-weight: bold;
    color: {PRIMARY_COLOR};
    margin-top: 5px;
    margin-bottom: 5px;
}}

.data-container {{
    background-color: {CARD_COLOR};
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}}

.stProgress > div > div > div > div {{
    background-color: {PRIMARY_COLOR};
}}
</style>
""", unsafe_allow_html=True)

# =========================
# URL DE LA API
# =========================
API_URL = "https://web-production-271f3.up.railway.app"  # ⚠️ Cambia por tu URL real si es distinta

# =========================
# OBTENER DATOS DE LA API
# =========================
try:
    response = requests.get(f"{API_URL}/metrics")
    data = response.json()
    st.success("✅ Conectado correctamente con la API")
except Exception as e:
    st.error("❌ No se pudo conectar con la API")
    st.exception(e)
    st.stop()

# =========================
# MÉTRICAS PRINCIPALES
# =========================
st.markdown(f"<h2>📈 Métricas del Modelo</h2>", unsafe_allow_html=True)

metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
cols = st.columns(len(metrics))

for col, name in zip(cols, metrics):
    with col:
        value = data.get(name, 0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(name, unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{value:.3f}</div>', unsafe_allow_html=True)
        st.progress(min(value, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MATRIZ DE CONFUSIÓN
# =========================
st.markdown(f"<h2>🧮 Matriz de Confusión</h2>", unsafe_allow_html=True)
cm = data.get("Confusion_Matrix", [[0, 0], [0, 0]])

fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG_COLOR)
sns.heatmap(cm, annot=True, fmt="d", cmap=HEATMAP_CMAP, cbar=False,
            xticklabels=["Pred. No", "Pred. Sí"],
            yticklabels=["Real No", "Real Sí"], ax=ax)
ax.set_xlabel("Predicho", color=TEXT_COLOR)
ax.set_ylabel("Real", color=TEXT_COLOR)
ax.set_title("Matriz de Confusión", color=PRIMARY_COLOR)
ax.tick_params(colors=TEXT_COLOR)
fig.patch.set_facecolor(BG_COLOR)
st.pyplot(fig)

# =========================
# GRÁFICAS DEL MODELO
# =========================
st.markdown(f"<h2>📉 Curvas de Rendimiento</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.image(f"{API_URL}/plot/roc", caption="Curva ROC", use_container_width=True)

with col2:
    st.image(f"{API_URL}/plot/precision_recall", caption="Curva Precision-Recall", use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:{TEXT_COLOR};'>Desarrollado por <b>Josué Alexander Saucedo González</b></p>",
    unsafe_allow_html=True
)


