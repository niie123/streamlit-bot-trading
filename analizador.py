import cv2
import pytesseract
import numpy as np
import re
import os

# Ruta al ejecutable de Tesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\00082484\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# 🔧 Escala global para redimensionar la imagen (40%)
ESCALA = 0.4

def cargar_imagen(ruta):
    original = cv2.imread(ruta)
    if original is None:
        print("❌ No se pudo cargar la imagen.")
        return None, None
    reducida = cv2.resize(original, (0, 0), fx=ESCALA, fy=ESCALA)
    return original, reducida

def convertir_coord(x, y):
    return int(x / ESCALA), int(y / ESCALA)

def detectar_precio_con_color(imagen, y1, y2, x1, x2, hsv_min, hsv_max):
    zona = imagen[y1:y2, x1:x2]
    hsv = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if 30 < w < 200 and 15 < h < 100:
            recorte = zona[y:y+h, x:x+w]
            gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            mejorado = cv2.equalizeHist(gris)
            _, binaria = cv2.threshold(mejorado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            texto_precio = pytesseract.image_to_string(
                binaria,
                config='--psm 7 -c tessedit_char_whitelist=0123456789.'
            )
            print("🧾 Texto Precio detectado:", texto_precio.strip())

            for token in texto_precio.split():
                try:
                    limpio = token.replace(",", "").replace("¢", "").replace("O", "0").replace("S", "5").replace("B", "8")
                    # Si es todo dígitos y largo, intenta inferir el punto decimal
                    if limpio.isdigit() and len(limpio) >= 5:
                        corregido = float(limpio[:-2] + "." + limpio[-2:])
                    else:
                        corregido = float(limpio)

                    if 0.5 < corregido < 1000000:
                        return corregido
                except:
                    continue

    return None


def analizar_imagen_con_recortes(ruta_imagen):
    img, img_reducida = cargar_imagen(ruta_imagen)
    if img is None:
        return

    # === Recortes RSI y par ===
    zona_rsi = img[2042:2107, 7:242]
    zona_par = img[302:367, 7:225]
    texto_rsi = pytesseract.image_to_string(zona_rsi)
    texto_par = pytesseract.image_to_string(zona_par)
    print("🧾 Texto RSI:", texto_rsi.strip())
    print("🧾 Texto Par/Temporalidad:", texto_par.strip())

    # === RSI ===
    rsi = None
    for token in texto_rsi.split():
        try:
            if token.replace('.', '', 1).isdigit():
                rsi = float(token)
                break
        except:
            continue

    # === MACD ===
    zona_macd = img[1260:1310, 12:610]
    gris_macd = cv2.cvtColor(zona_macd, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gris_macd)
    _, bin_macd = cv2.threshold(eq, 130, 255, cv2.THRESH_BINARY)
    texto_macd = pytesseract.image_to_string(bin_macd, config='--psm 6')
    print("🧾 OCR MACD crudo:", texto_macd.strip())
    texto_macd = texto_macd.replace('\n', ' ').replace('–', '-').replace(':', '.').replace('O', '0').replace('UID', '12')
    nums = re.findall(r'-?\d+\.\d+', texto_macd)
    macd_val = float(nums[0]) if len(nums) > 0 else None
    signal_val = float(nums[1]) if len(nums) > 1 else None

    # === PRECIO (zona definida por el usuario) ===
    y1, y2 = 377, 1187
    x1, x2 = 1155, 1317
    precio = detectar_precio_con_color(img, y1, y2, x1, x2, np.array([20, 100, 100]), np.array([35, 255, 255]))  # amarillo

    # === EMAs ===
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])) | cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))
    blue_mask = cv2.inRange(hsv, np.array([100,100,100]), np.array([130,255,255]))
    red_coords = np.column_stack(np.where(red_mask > 0))
    blue_coords = np.column_stack(np.where(blue_mask > 0))
    avg_red_y = np.mean(red_coords[:,0]) if red_coords.size > 0 else None
    avg_blue_y = np.mean(blue_coords[:,0]) if blue_coords.size > 0 else None

    # === ANÁLISIS ===
    print("\n📊 ANÁLISIS TÉCNICO")
    if rsi:
        print(f"✅ RSI detectado: {rsi}")
        if rsi < 30:
            print("🟢 RSI en sobreventa → posible COMPRA")
        elif rsi > 70:
            print("🔴 RSI en sobrecompra → posible VENTA")
        elif rsi < 40:
            print("🔻 RSI bajista")
        elif rsi > 60:
            print("🔺 RSI alcista")
        else:
            print("🟡 RSI neutral")
    else:
        print("❓ RSI no detectado.")

    if macd_val is not None and signal_val is not None:
        print(f"✅ MACD: {macd_val}, Señal: {signal_val}")
        if abs(macd_val - signal_val) < 0.05:
            print("🔄 MACD y Señal cercanos → Consolidación")
        elif macd_val > signal_val:
            print("📈 MACD alcista")
        else:
            print("📉 MACD bajista")
    else:
        print("❓ MACD no detectado correctamente.")

    if avg_blue_y and avg_red_y:
        print(f"🔵 EMA50 (azul): Y promedio = {avg_blue_y:.1f}")
        print(f"🔴 EMA200 (roja): Y promedio = {avg_red_y:.1f}")
        if avg_blue_y < avg_red_y:
            print("✅ EMA50 sobre EMA200 → Golden Cross (alcista)")
        else:
            print("⚠️ EMA50 bajo EMA200 → Death Cross (bajista)")
    else:
        print("❓ EMAs no detectadas con precisión.")

    if precio:
        print(f"💰 Precio actual detectado: {precio}")

        # Calculamos un margen dinámico (por ejemplo 0.5% del precio)
        margen = precio * 0.005

        zona_baja = precio - margen
        zona_alta = precio + margen

        print(f"📐 Margen dinámico aplicado: ±{margen:.2f}")
        print(f"📌 Zonas: baja < {zona_baja:.2f}, media entre {zona_baja:.2f} y {zona_alta:.2f}, alta > {zona_alta:.2f}")

        # Ahora comparamos el precio con esas zonas
        if precio < zona_baja:
            print("📉 Precio en zona baja (posible soporte)")
        elif precio > zona_alta:
            print("📈 Precio en zona alta (posible resistencia)")
        else:
            print("📊 Precio en zona media")
    else:
        print("❓ Precio actual no detectado.")

    # === Recomendación Final ===
    print("\n📌 Recomendación general:")
    if all([rsi, macd_val is not None, avg_blue_y, avg_red_y, precio]):
        if rsi < 30 and macd_val > signal_val and avg_blue_y < avg_red_y:
            print("🟢 Señales alineadas para posible COMPRA")
        elif rsi > 70 and macd_val < signal_val and avg_blue_y > avg_red_y:
            print("🔴 Señales alineadas para posible VENTA")
        else:
            print("🕒 Señales mixtas → esperar confirmación")
    else:
        print("🔍 Datos incompletos → revisar imagen o recortes")

# Ejecutar análisis
#analizar_imagen_con_recortes("image.jpg")
