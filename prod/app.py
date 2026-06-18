import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from utils import load_model_and_centroids, load_mtcnn, predict_faces

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Reconocimiento Facial — Selección Argentina 2022",
    page_icon="⚽",
    layout="wide"
)

# Detectar dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ruta relativa del checkpoint (ejecutado desde la carpeta prod/)
MODEL_PATH = Path(__file__).parent / "../dev/modelo.pth"

# ── Barra Lateral ──────────────────────────────────────────────
st.sidebar.title("Configuración")
st.sidebar.markdown("Ajustá los parámetros del modelo en tiempo real:")

# Slider interactivo para el umbral de rechazo
threshold = st.sidebar.slider(
    "Umbral de rechazo (Similitud Mínima)",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.05,
    help="Si la similitud del rostro con el jugador más cercano es menor a este valor, se clasificará como 'Desconocido'."
)

st.sidebar.divider()

# Lista de jugadores registrados (obtenida del checkpoint para no hardcodear)
if MODEL_PATH.exists():
    try:
        # Cargamos los recursos en memoria (cacheado para velocidad)
        model, centroids, idx_to_label = load_model_and_centroids(str(MODEL_PATH), DEVICE)
        mtcnn = load_mtcnn(DEVICE)
        
        st.sidebar.subheader("Jugadores Registrados (26)")
        players_list = sorted(list(idx_to_label.values()))
        # Mostrar en una lista colapsable (expander)
        with st.sidebar.expander("Ver lista de jugadores"):
            for p in players_list:
                st.markdown(f"- {p.replace('_', ' ')}")
    except Exception as e:
        st.sidebar.error(f"Error al cargar recursos: {e}")
else:
    st.sidebar.error(f"No se encontró el archivo del modelo en {MODEL_PATH.resolve()}")

# ── Contenido Principal ────────────────────────────────────────
st.title("⚽ Reconocimiento Facial — Selección Argentina FIFA 2022")
st.markdown("""
Subí una foto que contenga uno o varios rostros. El sistema los detectará automáticamente y buscará si coinciden con alguno de los **26 campeones del mundo de la selección argentina** en Qatar 2022.
""")

# Selección de método de entrada (Uploader o URL)
tab_upload, tab_url = st.tabs(["📂 Subir Imagen", "🔗 Ingresar URL"])

image = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Seleccioná una imagen (PNG, JPG, JPEG, WEBP):", 
        type=["png", "jpg", "jpeg", "webp"]
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error al abrir la imagen: {e}")

with tab_url:
    url_input = st.text_input("Ingresá la URL de la imagen:")
    if url_input:
        try:
            import requests
            from io import BytesIO
            
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url_input, headers=headers, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Error al descargar la imagen de la URL: {e}")

if image is not None:
    try:
        # Inferencia
        with st.spinner("Procesando imagen y buscando rostros..."):
            predictions = predict_faces(
                image, model, mtcnn, centroids, idx_to_label, threshold, DEVICE
            )
            
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Imagen Procesada")
            if not predictions:
                st.image(image, use_container_width=True)
                st.warning("⚠️ No se detectó ningún rostro en la imagen. Por favor, intentá con otra foto.")
            else:
                # Dibujar recuadros en la imagen original
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                for idx, pred in enumerate(predictions):
                    box = pred['box']
                    # Verde si es reconocido, Rojo si es desconocido
                    color = "#00FF00" if pred['is_recognized'] else "#FF0000"
                    draw.rectangle(box, outline=color, width=4)
                    
                    # Dibujar etiqueta de texto sobre el rostro
                    x1, y1, _, _ = box
                    label_text = pred['label'].replace('_', ' ') if pred['is_recognized'] else "Desconocido"
                    
                    # Estimar el ancho del fondo de la etiqueta (aprox 7px por caracter + margen)
                    text_width = len(label_text) * 7 + 10
                    
                    draw.rectangle([x1, y1 - 20, x1 + text_width, y1], fill=color)
                    draw.text((x1 + 5, y1 - 18), label_text, fill="#000000" if pred['is_recognized'] else "#FFFFFF")
                
                st.image(annotated_image, use_container_width=True)
                
        with col2:
            if predictions:
                st.subheader(f"Rostros Detectados ({len(predictions)})")
                
                # Mostrar en cuadricula de columnas
                cols_per_row = 3
                for i in range(0, len(predictions), cols_per_row):
                    cols = st.columns(min(cols_per_row, len(predictions) - i))
                    for j, col in enumerate(cols):
                        pred = predictions[i + j]
                        with col:
                            # Limitar el ancho de la cara para que no se vea gigante y estirada
                            st.image(pred['face_image'], caption=f"Rostro {i+j+1}", width=120)
                            
                            if pred['is_recognized']:
                                display_name = pred['label'].replace('_', ' ')
                                st.success(f"**{display_name}**")
                                st.markdown(f"Confianza: `{pred['confidence']:.2%}`")
                            else:
                                most_similar = pred['most_similar_player'].replace('_', ' ')
                                st.error(f"**Desconocido**")
                                st.markdown(f"Más similar a: **{most_similar}**")
                                st.markdown(f"Similitud: `{pred['confidence']:.2%}`")
                                
    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")
else:
    # Estado vacío por defecto
    st.info("Subí una imagen o ingresá una URL arriba para comenzar.")
    
    # Mostrar imágenes ejemplo sugeridas
    st.divider()
    st.markdown("### ¿Cómo funciona?")
    st.markdown("""
    1. **Detección:** Se usa un detector de rostros **MTCNN** para aislar cada rostro presente en la foto.
    2. **Embedding:** Cada cara recortada se pasa por el backbone de **FaceNet** entrenado para obtener un vector de características de 128 dimensiones.
    3. **Clasificación:** Se calcula la **similitud coseno** con los centroides promedio de cada jugador guardados durante el entrenamiento.
    4. **Umbral de rechazo:** Si el valor máximo de similitud no supera el límite configurado, se rechaza y se marca como **Desconocido** (evitando falsos positivos).
    """)
