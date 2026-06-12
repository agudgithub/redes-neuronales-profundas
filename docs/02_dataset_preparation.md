# Notebook 02 — Preparación del Dataset

**Archivo:** `dev/02_dataset_preparation.ipynb`  
**Objetivo:** Construir el dataset de imágenes de rostros de 26 futbolistas argentinos del Mundial FIFA 2022, listos para entrenar un clasificador con PyTorch.

---

## Tabla de contenidos

1. [Contexto general](#1-contexto-general)
2. [Paso 0 — Instalación de dependencias](#2-paso-0--instalación-de-dependencias)
3. [Paso 1 — Obtener el dataset original](#3-paso-1--obtener-el-dataset-original)
4. [Paso 2 — Extracción de rostros con InsightFace](#4-paso-2--extracción-de-rostros-con-insightface)
5. [Paso 3 — Construcción del DataFrame de Argentina](#5-paso-3--construcción-del-dataframe-de-argentina)
6. [Paso 4 — Split estratificado 60 / 20 / 20](#6-paso-4--split-estratificado-60--20--20)
7. [Paso 5 — FaceDataset y DataLoaders](#7-paso-5--facedataset-y-dataloaders)
8. [Compatibilidad con Google Colab](#8-compatibilidad-con-google-colab)

---

## 1. Contexto general

El dataset de partida es el **FIFA 2022 All Players Image Dataset** de Kaggle, que contiene fotografías de jugadores de todos los países participantes del mundial. El problema que resolvemos en este notebook es:

1. Filtrar sólo los **26 jugadores de la selección argentina**.
2. Detectar y recortar el **rostro** de cada imagen usando un modelo de detección neuronal.
3. Organizar los datos en un formato compatible con PyTorch.
4. Generar los splits de entrenamiento, validación y test de forma reproducible.

El resultado de este notebook son los archivos que consume el notebook 03:

```
data/
├── FIFA_2022_ONLY_FACES/      ← Imágenes de rostros (224×224)
│   └── Argentina Players/
│       └── Images_Jugador/
│           └── imagen.jpg
├── train.csv
├── val.csv
├── test.csv
└── label_to_idx.json
```

---

## 2. Paso 0 — Instalación de dependencias

```python
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "insightface", "onnxruntime-gpu", "gdown", ...])
else:
    # Fuerza onnxruntime-gpu para CUDA en Windows/Linux
    subprocess.check_call([...pip uninstall onnxruntime...])
    subprocess.check_call([...pip install onnxruntime-gpu...])
```

**¿Por qué dos ramas?**

- **En Colab** CUDA ya está configurado por la plataforma y un `pip install` estándar es suficiente.
- **En local (Windows)** el paquete `onnxruntime` (CPU) y `onnxruntime-gpu` coexisten con conflicto. Hay que desinstalar el CPU antes de instalar la versión GPU para que InsightFace use CUDA correctamente.

**¿Por qué `subprocess` en vez de `!pip install`?**

Los magic commands de Jupyter (`!pip`) son interpretados por el kernel de IPython y funcionan bien interactivamente. Pero `subprocess.check_call([sys.executable, "-m", "pip", ...])` garantiza que se use el *mismo intérprete de Python* que ejecuta el notebook, lo cual es más robusto al ejecutar el notebook como script o en entornos gestionados.

---

## 3. Paso 1 — Obtener el dataset original

```python
SKIP_EXTRACTION = True   # ← En Colab SIEMPRE True
DRIVE_ZIP_ID    = "1pG1qYiYy2p_dVOTuP2XjW2P430ykq6l-"
```

La lógica de descarga es la siguiente (en orden):

```python
if not Path(FACES_DIR).exists():
    if SKIP_EXTRACTION:
        # Descargar ZIP de caras desde Google Drive
        gdown.download(id=DRIVE_ZIP_ID, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
    else:
        # Descargar dataset completo desde Kaggle
        kagglehub.dataset_download("soumendraprasad/fifa-2022-all-players-image-dataset")
else:
    # Las caras ya existen — no se descarga nada
    pass
```

### `SKIP_EXTRACTION = True` — Flujo Colab / corridas repetidas *(recomendado)*

Si `FIFA_2022_ONLY_FACES/` no existe en disco, descarga el ZIP de rostros **ya procesados** desde Google Drive y lo extrae. Si ya existe, lo reutiliza directamente sin hacer ninguna descarga.

**¿Por qué?**  
La extracción con InsightFace (Paso 2) tarda ~30 segundos con GPU. Si las caras ya fueron extraídas y guardadas en Drive, no tiene sentido repetir ese proceso. Descargar el ZIP procesado es mucho más rápido (~10 segundos). El check de existencia evita descargas redundantes en corridas sucesivas dentro de la misma sesión de Colab.

### `SKIP_EXTRACTION = False` — Primera corrida en local

Descarga el dataset completo de Kaggle (~361 MB) usando `kagglehub`:

```python
os.environ["KAGGLE_API_TOKEN"] = token  # token en memoria, sin crear ~/.kaggle/
downloaded_path = kagglehub.dataset_download("soumendraprasad/fifa-2022-all-players-image-dataset")
```

**Decisión de diseño — token en memoria, no en disco:**  
`kagglehub` admite autenticación vía variable de entorno `KAGGLE_API_TOKEN`, evitando crear el archivo `~/.kaggle/kaggle.json` en el sistema del usuario. Esto es importante en Colab donde el directorio home es efímero y en producción donde no queremos credenciales en disco.

**Decisión de diseño — cache temporal dentro del proyecto:**  
`kagglehub` descarga a un cache global en `~/.cache/`. Para poder limpiarlo fácilmente y evitar que ocupe espacio en carpetas del sistema, redirigimos el cache al propio proyecto con `os.environ["KAGGLEHUB_CACHE"]` y lo eliminamos tras mover los archivos.

---

## 4. Paso 2 — Extracción de rostros con InsightFace

Este es el paso más importante del pipeline de datos. Se usa el modelo **`buffalo_l`** de [InsightFace](https://github.com/deepinsight/insightface), un detector de rostros de alta precisión con soporte GPU vía ONNX Runtime.

### Detección automática GPU / CPU

```python
available = ort.get_available_providers()
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available
    else ["CPUExecutionProvider"]
)
app = FaceAnalysis(providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))
```

ONNX Runtime puede correr en CPU o GPU. El código detecta automáticamente cuál está disponible y prioriza GPU. En una RTX 3070, la detección de 1300 imágenes tarda **~27 segundos**; en CPU tardaría ~5 minutos.

### Filtro de fotos grupales (heurística de área)

```python
if len(faces_with_areas) > 1:
    largest_area, second_area = ...
    if largest_area < 2.5 * second_area:
        return None  # Descartar — foto grupal ambigua
```

**¿Por qué?**  
El dataset original contiene fotos de festejos grupales y celebraciones donde aparecen varios jugadores. Entrenar con esas imágenes introduciría ruido: el modelo aprendería la imagen de un jugador etiquetada con otro. La heurística descarta cualquier imagen donde el segundo rostro detectado tiene más del 40% del área del primero (ratio < 2.5×), lo que indica que es una foto con varios protagonistas visibles.

### Corrección de encoding de nombres de carpetas

```python
def fix_encoding(name):
    try:
        return name.encode("cp866").decode("utf-8")
    except:
        try:
            return name.encode("cp437").decode("utf-8")
        ...
```

**¿Por qué?**  
Al descomprimir un ZIP en Windows, los nombres de carpeta con caracteres especiales (`Julián Álvarez`, `Ángel Di María`) pueden quedar mal decodificados dependiendo de la consola y el sistema operativo. La función `fix_encoding` intenta las codificaciones más comunes de Windows hasta encontrar la correcta, o devuelve el nombre sin cambios si ya está en UTF-8 (caso Linux/Mac/Colab).

### Procesamiento paralelo con ThreadPoolExecutor

```python
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_file, f): f for f in img_files}
    for fut in tqdm(as_completed(futures), ...):
        ...
```

**¿Por qué paralelo?**  
La detección con InsightFace (que corre en GPU) es mucho más rápida que la lectura de imágenes desde disco (I/O). Usando múltiples threads (`MAX_WORKERS=4`), se solapan la lectura de disco y la inferencia GPU, reduciendo el cuello de botella de I/O.

### Resultado de la extracción

De las 1300 imágenes originales de Argentina, se extraen y guardan **~1050-1270 rostros** válidos (dependiendo de la corrida). Las imágenes descartadas son fotos grupales, imágenes borrosas o poses donde InsightFace no detectó ningún rostro.

---

## 5. Paso 3 — Construcción del DataFrame de Argentina

```python
for img_path in faces_root.rglob("*"):
    rel_path = img_path.relative_to(faces_root)
    parts = rel_path.parts
    
    is_argentina = any("argentina" in p.lower() for p in parts)
    if not is_argentina:
        continue
    
    player_dir_name = parts[-2]
    player_name = player_dir_name.replace("Images_", "")
    
    records.append({
        "image_path": str(rel_path),
        "label": fix_encoding(player_name),
    })
```

La estructura de carpetas del dataset original es:
```
FIFA_2022_ONLY_FACES/
└── Argentina Players/
    └── Images_<NombreJugador>/
        └── imagen.jpg
```

El código extrae el nombre del jugador del nombre de la carpeta (`parts[-2]`), elimina el prefijo `Images_` y aplica la corrección de encoding.

### Filtro de jugadores con pocas imágenes

```python
counts = df["label"].value_counts()
df = df[df["label"].isin(counts[counts >= 5].index)]
```

Se descartan jugadores con menos de 5 imágenes para garantizar que el split estratificado pueda funcionar (mínimo 1 imagen por split por clase).

### Mapeo label → índice

```python
label_to_idx = {lbl: i for i, lbl in enumerate(sorted(all_labels))}
```

Se crea un mapeo determinístico (ordenado alfabéticamente) entre el nombre del jugador y un índice entero. Este mapeo se guarda como `label_to_idx.json` y es consumido por el notebook 03 para reconstruir los nombres de los jugadores durante la evaluación.

**Resultado final:** 26 jugadores con entre 34 y 50 imágenes cada uno (total ~1270 imágenes).

---

## 6. Paso 4 — Split estratificado 60 / 20 / 20

```python
train_df, temp_df = train_test_split(
    df, test_size=0.40, stratify=df["label_idx"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label_idx"], random_state=42
)
```

| Split | Imágenes | Propósito |
|-------|----------|-----------|
| **Train** | ~761 (60%) | Ajuste de pesos del modelo |
| **Val** | ~254 (20%) | Selección de hiperparámetros y early stopping |
| **Test** | ~254 (20%) | Evaluación final imparcial (se toca **una sola vez**) |

**¿Por qué estratificado?**  
`stratify=df["label_idx"]` garantiza que cada jugador tiene exactamente el mismo porcentaje de imágenes en cada split. Sin esto, podría ocurrir que Messi tenga el 80% de sus imágenes en train y Foyth (con menos fotos) sólo el 40%, sesgando las métricas de evaluación.

**¿Por qué `random_state=42`?**  
Reproducibilidad. Cualquier persona que ejecute el notebook en cualquier máquina obtendrá exactamente el mismo split.

---

## 7. Paso 5 — FaceDataset y DataLoaders

```python
class FaceDataset(Dataset):
    def __init__(self, dataframe, root_dir,
                 use_flip=False, use_color_jitter=False,
                 use_rotation=False, use_erasing=False):
        steps = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if use_flip:
            steps.append(transforms.RandomHorizontalFlip(p=0.5))
        if use_color_jitter:
            steps.append(transforms.ColorJitter(...))
        if use_rotation:
            steps.append(transforms.RandomRotation(degrees=10))
        steps += [transforms.ToTensor(), transforms.Normalize(...)]
        if use_erasing:
            steps.append(transforms.RandomErasing(...))
```

**¿Por qué augmentaciones como flags individuales?**  
El notebook 03 usa tres experimentos con diferentes niveles de augmentación (ninguna → moderada → agresiva). Tener cada augmentación como un booleano independiente permite construir el pipeline de transformaciones exactamente como se necesita en cada experimento, sin duplicar código.

**¿Por qué `NUM_WORKERS = 0`?**  
En Windows, el multiprocessing de PyTorch (`num_workers > 0`) requiere que el código esté protegido por `if __name__ == '__main__'`, lo cual no aplica en notebooks de Jupyter. Con `num_workers=0` toda la carga de datos corre en el proceso principal, evitando errores de multiprocessing en Windows sin sacrificar velocidad significativa con datasets pequeños (~1270 imágenes).

---

## 8. Compatibilidad con Google Colab

El notebook detecta el entorno automáticamente y ajusta las rutas:

```python
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    DATA_DIR    = "/content/data"
    FACES_DIR   = "/content/data/FIFA_2022_ONLY_FACES"
else:
    DATA_DIR    = "../data"
    FACES_DIR   = "../data/FIFA_2022_ONLY_FACES"
```

### Flujo recomendado en Colab

1. Abrir el notebook en Colab.
2. Asegurarse de que `SKIP_EXTRACTION = True` (valor por defecto).
3. Ejecutar todo el notebook (*Run All*).
4. El notebook detecta automáticamente que `FIFA_2022_ONLY_FACES/` no existe y descarga el ZIP desde Google Drive, lo extrae en `/content/data/`, construye el DataFrame y genera los CSVs.
5. Con los CSVs generados, abrir el notebook 03 en Colab y ejecutarlo.

> **Nota:** si se ejecuta la sesión varias veces sin reiniciar el runtime, el check de existencia de `FACES_DIR` evita que el ZIP se descargue de nuevo.
