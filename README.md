# ⚽ Reconocimiento Facial — Selección Argentina FIFA 2022

Este es el repositorio del **Trabajo Práctico Integrador** para la materia **Redes Neuronales Profundas** (Ingeniería en Sistemas de Información). 

Consiste en una aplicación web interactiva que detecta y reconoce los rostros de los **26 jugadores campeones del mundo de la Selección Argentina** en el Mundial de Qatar 2022 a partir de cualquier imagen cargada, utilizando técnicas modernas de visión por computadora y aprendizaje profundo.

---

## 🔗 Enlaces del Proyecto

* **Aplicación Web Desplegada:** [https://reconocimiento-facial-seleccion-argentina.streamlit.app/](https://reconocimiento-facial-seleccion-argentina.streamlit.app/)
* **Dataset de Origen (Kaggle):** [FIFA 2022 All Players Image Dataset](https://www.kaggle.com/datasets/soumendraprasad/fifa-2022-all-players-image-dataset)

---

## 👥 Integrantes
* **Ivan Duga** (GitHub: [@agudgithub](https://github.com/agudgithub))
* **Facundo Aracena**
* **Renzo Ortiz**
* **Marcio Palazzo**
* **Franco Sorrentino**

---

## 🛠️ Arquitectura del Componente Inteligente

El sistema de reconocimiento facial utiliza una arquitectura robusta de **Joint Embedding (Aprendizaje de Métricas)**:
1. **Detección y Recorte de Rostros (MTCNN):** Localiza y recorta de manera automática todos los rostros presentes en la imagen de entrada (soporta fotos grupales y de cuerpo entero).
2. **Backbone Feature Extractor (FaceNet):** Red neuronal convolucional `InceptionResnetV1` preentrenada en `VGGFace2`, encargada de mapear el rostro recortado a un espacio latente de alta dimensionalidad.
3. **Proyección (Joint Embedding Net):** Una cabeza lineal que proyecta el embedding a 128 dimensiones con normalización $L_2$.
4. **Clasificación por Centroides y Umbral:** Compara el embedding del rostro contra los centroides promedio de cada jugador mediante **similitud coseno**. Si la máxima similitud supera el **umbral de rechazo ajustable**, se reconoce al jugador; de lo contrario, se clasifica como **Desconocido**.

---

## 📂 Estructura del Repositorio

El repositorio está organizado siguiendo la estructura estándar requerida:

```text
redes-neuronales-profundas/
├── data/                       # Metadata del dataset (CSVs de splits y mapeos)
│   ├── label_to_idx.json
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── dev/                        # Desarrollo y experimentación (Jupyter Notebooks)
│   ├── 02_dataset_preparation.ipynb    # Descarga, detección (InsightFace) y preparación
│   ├── 03_model_training.ipynb         # Entrenamiento, refinamiento (R1-R3) y curvas
│   ├── 04_inference_prototyping.ipynb  # Prototipo de inferencia local con MTCNN y umbral
│   └── modelo.pth                      # Checkpoint unificado final (LFS)
├── docs/                       # Documentación del proyecto y enunciado del TP
│   ├── 02_dataset_preparation.md
│   ├── 03_model_training.md
│   └── TP_Integrador_2026.pdf
├── prod/                       # Código de producción de la aplicación web
│   ├── app.py                          # Interfaz interactiva de Streamlit
│   ├── utils.py                        # Funciones auxiliares y predicción optimizada
│   └── requirements.txt                # Dependencias para producción
├── .gitattributes              # Configuración de Git LFS para modelo.pth
└── README.md                   # Descripción general del proyecto
```

---

## 🚀 Ejecución en Local

Para clonar el proyecto y correr la aplicación de forma local, seguí estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/agudgithub/redes-neuronales-profundas.git
   cd redes-neuronales-profundas
   ```

2. **Crear e iniciar un entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   # En Windows (CMD):
   venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r prod/requirements.txt
   ```

4. **Ejecutar la aplicación Streamlit:**
   ```bash
   streamlit run prod/app.py
   ```
   A continuación, se abrirá automáticamente una pestaña en tu navegador en `http://localhost:8501`.

---

## ☁️ Despliegue en Streamlit Cloud

Para desplegar la aplicación de manera pública y gratuita:
1. Asegurate de subir todos tus cambios a GitHub (incluyendo el archivo `dev/modelo.pth` trackeado correctamente con Git LFS).
2. Entrá a [Streamlit Share](https://share.streamlit.io/) y vinculá tu cuenta de GitHub.
3. Hacé clic en **"New app"**, elegí el repositorio `redes-neuronales-profundas`, la rama `main` y seleccioná el archivo de entrada principal como `prod/app.py`.
4. Hacé clic en **"Deploy"** y Streamlit se encargará de instalar las dependencias y levantar el servidor.