import streamlit as st
from analizador import analizar_imagen_con_recortes
import os

st.title("📊 Análisis de Imagen de Trading")
st.write("Sube una imagen de un gráfico para analizar RSI, MACD, EMAs y precio actual.")

imagen = st.file_uploader("📷 Cargar imagen", type=["jpg", "jpeg", "png"])

if imagen:
    with open("image.jpg", "wb") as f:
        f.write(imagen.read())

    st.success("✅ Imagen cargada correctamente.")
    st.write("---")
    st.code("🔍 Análisis en consola:")

    # Ejecutar análisis y mostrar resultado en consola
    analizar_imagen_con_recortes("image.jpg")
