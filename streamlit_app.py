import streamlit as st
from analizador import analizar_imagen_con_recortes
import os

st.title("📊 Análisis de Imagen de Trading con IA")
st.write("Sube una imagen de un gráfico para analizar RSI, MACD, EMAs y detectar zona de velas japonesas.")

imagen = st.file_uploader("📷 Cargar imagen", type=["jpg", "jpeg", "png"])

if imagen:
    with open("image.jpg", "wb") as f:
        f.write(imagen.read())

    st.image("image.jpg", caption="📷 Imagen cargada", use_container_width=True)

    resultado = analizar_imagen_con_recortes("image.jpg")

    st.subheader("📌 Resultado del Análisis")
    st.text(resultado)

    if os.path.exists("recorte_velas.jpg"):
        st.image("recorte_velas.jpg", caption="🕯️ Zona de velas recortada", use_container_width=True)
