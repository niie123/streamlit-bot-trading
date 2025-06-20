import streamlit as st
from analizador import analizar_imagen_con_recortes
import os

st.set_page_config(page_title="Análisis de Trading con IA", layout="wide")

st.title("📊 Análisis de Imagen de Trading con IA")
st.write("Sube una imagen de un gráfico para analizar RSI, MACD, EMAs y detectar patrones técnicos con visión por computadora.")

#imagen = st.file_uploader("📷 Cargar imagen del gráfico", type=["jpg", "jpeg", "png"])

if imagen:
    with open("image.jpg", "wb") as f:
        f.write(imagen.read())

    st.image("image.jpg", caption="📷 Imagen cargada", use_container_width=True)

    with st.spinner("🔍 Analizando imagen..."):
        resultado = analizar_imagen_con_recortes("image.jpg")

    st.subheader("📌 Resultado del Análisis")
    st.text(resultado)

    if os.path.exists("recorte_velas.jpg"):
        st.image("recorte_velas.jpg", caption="🕯️ Zona de velas recortada", use_container_width=True)

    if os.path.exists("runs/detect/predict/image0.jpg"):
        st.image("runs/detect/predict/image0.jpg", caption="📸 Patrón detectado con IA (YOLOv8)", use_container_width=True)
