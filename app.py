import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from collections import deque

# --- CONFIGURACIÓN DE PALETAS (BGR para OpenCV) ---
PALETAS = {
    "INVIERNO": [
        {"bgr": (255, 0, 0), "n": "Azul Real"}, {"bgr": (130, 0, 75), "n": "Magenta"},
        {"bgr": (255, 255, 255), "n": "Blanco Puro"}, {"bgr": (0, 0, 0), "n": "Negro"}
    ],
    "VERANO": [
        {"bgr": (200, 150, 150), "n": "Azul Pastel"}, {"bgr": (180, 180, 210), "n": "Lavanda"},
        {"bgr": (170, 170, 170), "n": "Gris Perla"}, {"bgr": (180, 130, 150), "n": "Rosa"}
    ],
    "CALIDO": [
        {"bgr": (0, 165, 255), "n": "Naranja"}, {"bgr": (0, 128, 128), "n": "Verde Oliva"},
        {"bgr": (0, 75, 150), "n": "Terracota"}, {"bgr": (0, 190, 255), "n": "Ambar"}
    ],
    "NEUTRO": [
        {"bgr": (34, 34, 178), "n": "Rojo"}, {"bgr": (128, 0, 0), "n": "Azul Marino"},
        {"bgr": (80, 128, 0), "n": "Esmeralda"}, {"bgr": (200, 220, 240), "n": "Hueso"}
    ]
}

class ColorProcessor(VideoProcessorBase):
    def __init__(self):
        # Suavizado de datos para evitar parpadeos en el móvil
        self.hist_b = deque(maxlen=15)
        self.hist_con = deque(maxlen=15)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Efecto espejo
        h, w, _ = img.shape

        # 1. Áreas de Muestreo (Proporcionales al tamaño de pantalla)
        s = int(w * 0.05)
        # Cuadro Piel (Centro)
        py1, py2, px1, px2 = h//2-s, h//2+s, w//2-s, w//2+s
        # Cuadro Cabello (Superior)
        cy1, cy2, cx1, cx2 = h//2-int(h*0.2), h//2-int(h*0.15), w//2-s, w//2+s

        roi_piel = img[py1:py2, px1:px2]
        roi_pelo = img[cy1:cy2, cx1:cx2]

        # 2. Análisis LAB (Color y Luminosidad)
        lab_piel = cv2.cvtColor(roi_piel, cv2.COLOR_BGR2LAB)
        lab_pelo = cv2.cvtColor(roi_pelo, cv2.COLOR_BGR2LAB)
        
        b_val = np.mean(lab_piel[:, :, 2])
        con_val = abs(np.mean(lab_piel[:, :, 0]) - np.mean(lab_pelo[:, :, 0]))

        self.hist_b.append(b_val)
        self.hist_con.append(con_val)
        
        avg_b = sum(self.hist_b) / len(self.hist_b)
        avg_con = sum(self.hist_con) / len(self.hist_con)

        # 3. Lógica de Estaciones
        msg_extra = ""
        if avg_b > 131:
            est, col_ui, recs = "CALIDO", (0, 165, 255), PALETAS["CALIDO"]
        elif avg_b < 125:
            if avg_con > 45:
                est, col_ui, recs = "INVIERNO", (255, 50, 50), PALETAS["INVIERNO"]
            else:
                est, col_ui, recs = "VERANO", (220, 180, 180), PALETAS["VERANO"]
        else:
            est, col_ui, recs = "NEUTRO", (0, 255, 0), PALETAS["NEUTRO"]
            msg_extra = "Casi todos los colores te quedan"

        # 4. Dibujar Interfaz (Letras Negras, Gruesas y Nítidas)
        cv2.rectangle(img, (px1, py1), (px2, py2), col_ui, 2)
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1)

        # Resultado Estación
        cv2.putText(img, f"ESTACION: {est}", (20, h-70), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        if msg_extra:
            cv2.putText(img, msg_extra, (20, h-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        # Dibujar Paleta Lateral
        for i, obj in enumerate(recs):
            y_pos = 40 + (i * 60)
            cv2.rectangle(img, (w-60, y_pos), (w-20, y_pos+45), obj["bgr"], -1)
            cv2.putText(img, obj["n"], (w-180, y_pos+30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# --- INTERFAZ WEB (STREAMLIT) ---
st.set_page_config(page_title="Colorimetría IA", layout="centered")

st.title("Asesor de Imagen Digital 📱")
st.markdown("""
### ¡Analiza tu tono en segundos!
1. Presiona **Start** abajo.
2. Alinea tu mejilla en el cuadro de color y tu cabello en el blanco.
3. El sistema te dirá tu estación ideal.
""")

# Configuración técnica para que funcione en cualquier red (STUN servers)
webrtc_streamer(
    key="analisis-color",
    video_processor_factory=ColorProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)