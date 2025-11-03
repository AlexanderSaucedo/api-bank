import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(page_title="Dashboard SVM", layout="wide")

st.title("📊 Dashboard - Modelo SVM (Bank Marketing)")

# =========================
# URL DE LA API (modifica según tu URL en Railway)
# =========================
API_URL = "https://web-production-271f3.up.railway.app/"  # ⚠️ cambia esto por tu endpoint real

# =========================
# OBTENER DATOS DE LA API
# =========================
try:
    response = requests.get(f"{API_URL}/metrics")
    data = response.json()
    st.success("✅ Conectado correctamente con la API")
except Exception as e:
    st.error("❌ No se pudo conectar con la API")
    st.stop()

# =========================
# MOSTRAR MÉTRICAS
# =========================
st.subheader("📈 Métricas del Modelo")

cols = st.columns(5)
metricas = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]

for i, m in enumerate(metricas):
    cols[i].metric(label=m, value=data[m])

# =========================
# MATRIZ DE CONFUSIÓN
# =========================
st.subheader("🧮 Matriz de Confusión")
cm = data["Confusion_Matrix"]

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred. No", "Pred. Sí"],
            yticklabels=["Real No", "Real Sí"])
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
st.pyplot(fig)

# =========================
# GRÁFICAS ROC / PRECISION-RECALL
# =========================
st.subheader("📉 Gráficas del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.image(f"{API_URL}/plot/roc", caption="Curva ROC")

with col2:
    st.image(f"{API_URL}/plot/precision_recall", caption="Curva Precision-Recall")

