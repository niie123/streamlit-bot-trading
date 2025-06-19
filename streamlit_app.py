import streamlit as st
from analizador import analizar_imagen_con_recortes

st.title("📊 Análisis de Imagen de Trading")
st.write("Sube una imagen de un gráfico para analizar RSI, MACD, EMAs y precio actual.")

imagen = st.file_uploader("📷 Cargar imagen", type=["jpg", "jpeg", "png"])

if imagen:
    with open("image.jpg", "wb") as f:
        f.write(imagen.read())
    
    # ✅ Se reemplazó use_column_width por use_container_width
    #st.image("image.jpg", caption="📷 Imagen cargada", use_container_width=True)
    
    st.success("✅ Imagen cargada correctamente.")
    st.write("---")

    resultado = analizar_imagen_con_recortes("image.jpg")
    
    st.subheader("🔍 Resultado del Análisis")
    st.text(resultado)  # O usa st.markdown(resultado) si tiene formato
