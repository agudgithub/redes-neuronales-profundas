# Notebook 03 — Entrenamiento y Selección del Mejor Modelo

**Archivo:** `dev/03_model_training.ipynb`  
**Objetivo:** Entrenar tres arquitecturas diferentes sobre el dataset de rostros de futbolistas argentinos, compararlas con mAP, seleccionar el modelo ganador y aplicarle estrategias de refinamiento.

---

## Tabla de contenidos

1. [Contexto general](#1-contexto-general)
2. [Setup y configuración global](#2-setup-y-configuración-global)
3. [Arquitectura compartida: JointEmbeddingNet](#3-arquitectura-compartida-jointembeddingnet)
4. [Función de pérdida combinada](#4-función-de-pérdida-combinada)
5. [Métrica principal: mAP](#5-métrica-principal-map)
6. [Experimento A — FaceNet (baseline)](#6-experimento-a--facenet-baseline)
7. [Experimento B — EfficientNet-B2 (fine-tuning parcial)](#7-experimento-b--efficientnet-b2-fine-tuning-parcial)
8. [Experimento C — MobileNetV3-Large (fine-tuning total)](#8-experimento-c--mobilenetv3-large-fine-tuning-total)
9. [Selección del ganador](#9-selección-del-ganador)
10. [Evaluación en test y visualización](#10-evaluación-en-test-y-visualización)
11. [Refinamiento del modelo ganador](#11-refinamiento-del-modelo-ganador)
12. [Compatibilidad con Google Colab](#12-compatibilidad-con-google-colab)

---

## 1. Contexto general

El notebook 03 recibe los archivos generados por el notebook 02 y construye un **clasificador de identidad facial** para los 26 jugadores de la selección argentina.

El problema es desafiante por varias razones:
- **Dataset pequeño**: ~30 imágenes por clase en entrenamiento.
- **Alta similaridad intra-clase**: los jugadores con gorros de portero, cascos o diferentes peinados varían mucho entre imágenes.
- **Alta similaridad inter-clase**: algunos jugadores se parecen físicamente.

La estrategia es comparar tres configuraciones de fine-tuning sobre backbones preentrenados, usando una arquitectura de embedding compartida.

---

## 2. Setup y configuración global

### Configuración del dispositivo

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Todo el código usa la variable global `DEVICE`. Todos los tensores y modelos se mueven a este dispositivo con `.to(DEVICE)`.

### Semilla reproducible

```python
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**¿Por qué todos estos seeds?**  
PyTorch, NumPy y Python tienen generadores de números aleatorios independientes. Para que experimentos sean reproducibles hay que fijar los tres. Además, `cudnn.deterministic=True` desactiva los algoritmos de convolución no determinísticos que usa CUDA por defecto para ser más rápido.

### Parámetros clave

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `IMG_SIZE` | 224 | Resolución estándar de ImageNet; todos los backbones la esperan |
| `BATCH_SIZE` | 32 | Equilibrio entre estabilidad del gradiente y velocidad |
| `NUM_WORKERS` | 0 | Evita errores de multiprocessing en Windows/Jupyter |
| `NUM_CLASSES` | 26 | Un jugador por clase |

---

## 3. Arquitectura compartida: JointEmbeddingNet

```python
class JointEmbeddingNet(nn.Module):
    def __init__(self, backbone, embedding_dim=128, num_classes=26):
        ...
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        logits = self.classifier_head(embeddings)
        return embeddings, logits
```

### ¿Por qué esta arquitectura de dos salidas?

La red produce simultáneamente **embeddings** (representación compacta del rostro en el espacio latente) y **logits** (puntuaciones de clasificación). Esto permite:

1. **Entrenamiento**: usar la pérdida de clasificación (CrossEntropy) + pérdida de métrica (Triplet Loss) conjuntamente.
2. **Inferencia**: clasificar via similitud de coseno con centroides en lugar del cabezal lineal, lo que es más robusto con pocas imágenes por clase.

### ¿Por qué normalización L2 en los embeddings?

```python
embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
```

Normalizar los embeddings a la esfera unitaria es esencial para la Triplet Loss y para el cálculo de similitud de coseno. Sin normalización, la magnitud del vector dominaría sobre la dirección, invalidando la métrica de similitud.

### ¿Por qué `embedding_dim=128`?

128 dimensiones es un balance entre:
- **Capacidad representacional**: suficiente para separar 26 identidades.
- **Regularización**: demasiadas dimensiones con pocas imágenes llevan a overfitting.
- **Velocidad**: un espacio de 128 dims es cómodo para el cálculo de centroides y similitudes.

---

## 4. Función de pérdida combinada

### CrossEntropy + Triplet Loss

```python
loss_ce = criterion_ce(logits, labels)
loss_triplet = batch_all_triplet_loss(embeddings, labels, margin=1.0)
loss = loss_ce + loss_triplet
```

**¿Por qué combinar dos pérdidas?**

- **CrossEntropy (CE)**: supervisa directamente la tarea de clasificación. Aprende a asignar la clase correcta con alta probabilidad softmax.
- **Triplet Loss (Batch-All)**: supervisa la **métrica** en el espacio latente. Aprende que imágenes del mismo jugador deben estar cerca (similitud alta) y de jugadores distintos lejos (similitud baja).

Usar sólo CE produce representaciones separables en el clasificador lineal pero no necesariamente bien organizadas en el espacio latente. Usar sólo Triplet Loss con pocas clases puede ser inestable. La combinación da lo mejor de ambos mundos.

### Batch-All Triplet Loss

```python
def batch_all_triplet_loss(embeddings, labels, margin=1.0):
    # Calcula distancias entre todos los pares del batch
    distances = 2.0 - 2.0 * torch.matmul(embeddings, embeddings.t())
    
    # Construye máscara de tripletas válidas (anchor, positive, negative)
    mask = mask_anchor_positive.unsqueeze(2) & mask_anchor_negative.unsqueeze(1)
    
    triplet_loss = clamp(dist_ap - dist_an + margin, min=0)
    loss = sum(active_triplets) / count(active_triplets)
```

En vez de muestrear tripletas individuales (Triplet Mining estándar), la variante **Batch-All** considera **todas las tripletas válidas** dentro del batch. Con batch_size=32 y 26 clases, hay ~1 imagen por clase por batch → las tripletas hard están naturalmente representadas.

**¿Por qué `margin=1.0`?**  
Los embeddings están normalizados a la esfera unitaria, por lo que las distancias están en el rango [0, 2]. Un margen de 1.0 es una holgura del 50% del rango máximo, que funciona bien empíricamente en este dominio.

### Clasificación por centroides en evaluación

```python
def compute_class_centroids(model, loader):
    # Promedia los embeddings de cada clase en el conjunto de train
    for c in range(NUM_CLASSES):
        class_embs = embeddings[labels == c]
        centroid = normalize(class_embs.mean(dim=0))
    return centroids

# En evaluación:
similarities = torch.matmul(embeddings, centroids.t())
logits = similarities * 10.0  # Factor de escala para softmax
```

**¿Por qué centroide en vez del cabezal lineal para evaluar?**  
Con ~30 imágenes de entrenamiento por clase, el cabezal lineal puede sobreajustarse. El centroide es el "prototipo" de cada jugador calculado sobre todos los embeddings disponibles. Es más robusto porque agrega información de todas las imágenes de entrenamiento.

El factor `× 10.0` escala la similitud coseno (rango [-1, 1]) a un rango más amplio antes del softmax, produciendo distribuciones de probabilidad más nítidas.

---

## 5. Métrica principal: mAP

```python
MulticlassAveragePrecision(num_classes=26, average='macro')
```

**¿Por qué mAP y no Accuracy?**

El **mAP (mean Average Precision)** es más informativo que el Accuracy en este contexto por dos razones:

1. **Captura el ranking de probabilidades**: si el modelo asigna 0.8 a la clase correcta versus 0.51, eso se ve reflejado en el AP. Dos modelos con el mismo Accuracy (acierto top-1) pueden tener mAP muy diferente si uno está bien calibrado y el otro no.

2. **Es más justo con clases desbalanceadas**: Messi tiene 46 imágenes y Pezzella tiene 47, pero 26 imágenes de Pezzella en test (los más populares tienden a tener más fotografías). El AP por clase captura el rendimiento individual antes de promediar.

**Fórmula:**

$$\text{mAP} = \frac{1}{26} \sum_{k=1}^{26} \text{AP}_k$$

Donde cada $\text{AP}_k$ es el área bajo la curva precision-recall del problema binario "¿es el jugador $k$ o no?"

Junto con mAP también se reportan **Accuracy macro** y **F1-macro** como métricas complementarias.

---

## 6. Experimento A — FaceNet (baseline)

### Arquitectura

- **Backbone**: `InceptionResnetV1(pretrained='vggface2')` — 23.5M parámetros **congelados**
- **Cabeza entrenada**: `embedding_layer` (512→128) + `classifier_head` (128→26) — ~70K parámetros
- **Augmentaciones**: ninguna

### ¿Por qué FaceNet como baseline?

FaceNet es el punto de partida ideal para reconocimiento facial porque:

- Es una arquitectura **especializada en rostros**, preentrenada en el dataset VGGFace2 con millones de caras de miles de identidades.
- Sus capas ya aprendieron micro-estructuras faciales (distancia interocular, forma de nariz, contorno facial), a diferencia de ResNet/EfficientNet que se preentrenaron con objetos genéricos de ImageNet.
- Proporciona un espacio latente de 512 dimensiones muy denso y discriminativo para similitud facial.

### ¿Por qué congelar el backbone?

Con el backbone congelado, el experimento responde a la pregunta: **¿son los pesos preentrenados de FaceNet suficientes para separar a los 26 jugadores, entrenando sólo la cabeza?**

Esto también permite una comparación limpia: si FaceNet frozen supera a EfficientNet fine-tuned, el dominio (rostros) importa más que la arquitectura.

### ¿Por qué Adam + ReduceLROnPlateau?

Con sólo ~70K parámetros entrenables, Adam converge muy rápido. `ReduceLROnPlateau(patience=3)` reduce el LR a la mitad cuando el mAP en validación no mejora en 3 épocas consecutivas, evitando que el optimizador oscile alrededor del mínimo sin progresar.

```python
criterion_a = nn.CrossEntropyLoss()
optimizer_a = torch.optim.Adam(
    list(model_a.embedding_layer.parameters()) + 
    list(model_a.classifier_head.parameters()),
    lr=1e-3
)
scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_a, mode='max', patience=3, factor=0.5
)
model_a, history_a, best_map_a = run_experiment(..., n_epochs=25)
```

---

## 7. Experimento B — EfficientNet-B2 (fine-tuning parcial)

### Arquitectura

- **Backbone**: `efficientnet_b2(weights=IMAGENET1K_V1)` — 9M parámetros
- **Fine-tuning parcial**: bloques `features[5..8]` descongelados (últimas 3 de 8)
- **Augmentaciones**: flip horizontal + color jitter moderado
- **LR diferencial**: backbone `5e-5`, cabeza `5e-4`
- **Label smoothing = 0.1**

### ¿Por qué EfficientNet-B2?

EfficientNet aplica **compound scaling**: escala simultáneamente profundidad, ancho y resolución con un coeficiente único, produciendo mejor relación accuracy/parámetros que arquitecturas que solo escalan en una dimensión.

- **9M parámetros** vs 25M de FaceNet → 3× más compacto (~36 MB como `.pth`, bajo el límite de GitHub sin Git LFS).
- Accuracy en ImageNet comparable a FaceNet con menos parámetros → mayor regularización implícita.

### ¿Por qué congelar los primeros 5 bloques?

En EfficientNet, `features[0..4]` detecta edges, texturas y formas simples — features universales que transfieren bien a cualquier dominio visual. `features[5..8]` captura semántica de alto nivel específica de las 1000 clases de ImageNet y necesita adaptarse para distinguir 26 caras de futbolistas.

Esta división sigue la heurística de *descongelar desde arriba hacia abajo*:
- **Adaptación**: las últimas capas deben aprender a separar estas 26 caras específicas.
- **Estabilidad**: descongelar todo con ~28 imágenes/clase en train causaría olvido catastrófico.

### ¿Por qué LR diferenciales (discriminative fine-tuning)?

Aplicar el mismo LR a capas preentrenadas y a la nueva cabeza genera una tensión: la cabeza necesita un LR alto para aprender desde cero, pero el backbone preentrenado se degradaría con ese mismo LR. La solución (Howard & Ruder, ULMFiT 2018):

```python
optimizer_b = torch.optim.AdamW([
    {'params': backbone_params_b, 'lr': 5e-5, 'weight_decay': 1e-4},  # 10× más lento
    {'params': head_params_b,     'lr': 5e-4, 'weight_decay': 1e-4},
])
```

**AdamW** agrega la penalización L2 directamente sobre los pesos (no sobre el gradiente como Adam clásico) → mejor regularización durante fine-tuning.

### ¿Por qué label_smoothing = 0.1?

Con ~28 imágenes/clase, el modelo tiende a ser sobreconfiante (probabilidad cercana a 1.0 para la clase correcta). Esto degrada el mAP: la curva precision-recall se "aplana" en valores altos de confianza. `label_smoothing=0.1` distribuye 10% de la masa de probabilidad a las otras 25 clases → mejor calibración → mejora directa en mAP.

### ¿Por qué CosineAnnealingLR?

```python
scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=30)
```

Baja el LR siguiendo una curva coseno desde `lr_max` hasta ~0 en 30 épocas, sin discontinuidades como el step decay. Complementa bien AdamW porque la penalización L2 y la reducción suave del LR actúan juntas como regularización.

---

## 8. Experimento C — MobileNetV3-Large (fine-tuning total)

### Arquitectura

- **Backbone**: `mobilenet_v3_large(weights=IMAGENET1K_V2)` — 5M parámetros
- **Fine-tuning**: TODO el backbone descongelado desde el inicio
- **Augmentaciones**: flip + color jitter + rotación + random erasing (agresivas)
- **LR diferencial** con CosineAnnealingLR

### ¿Por qué MobileNetV3?

MobileNetV3-Large fue diseñado para **inferencia en dispositivos móviles**: usa bloques Inverted Residuals y un módulo de atención H-Swish que lo hace eficiente en parámetros. Con sólo 5M parámetros, el riesgo de overfitting es menor que con EfficientNet, por lo que se puede descongelar todo el backbone.

### ¿Por qué fine-tuning completo?

Con MobileNetV3 se testea el extremo opuesto al Experimento A: **¿qué pasa si adaptamos todos los pesos al dataset de futbolistas?** Con augmentaciones agresivas para compensar el overfitting potencial.

### ¿Por qué augmentaciones más agresivas?

Al descongelar todos los parámetros con pocas imágenes por clase, el riesgo de overfitting es mayor. Las augmentaciones actúan como regularización implícita:

| Augmentación | Efecto |
|-------------|--------|
| `RandomHorizontalFlip(p=0.5)` | Aprende simetría facial |
| `ColorJitter(0.2, 0.2, 0.2)` | Invarianza a condiciones de iluminación |
| `RandomRotation(10°)` | Invarianza a pequeñas rotaciones de cabeza |
| `RandomErasing(p=0.2, scale=(0.02, 0.15))` | Simula oclusiones parciales (pelo, mano, micrófono) |

### ¿Por qué IMAGENET1K_V2?

Los pesos `V2` de MobileNetV3 fueron entrenados con recetas modernas (mixup, augmentaciones en training, EMA de pesos) produciendo mejores representaciones que los pesos V1 originales, especialmente en fine-tuning.

---

## 9. Selección del ganador

```python
candidates = [
    ('A', model_a, best_map_a),
    ('B', model_b, best_map_b),
    ('C', model_c, best_map_c),
]
winner_name, best_model, winner_map = max(candidates, key=lambda x: x[2])
```

La selección se basa exclusivamente en el **mejor mAP en validación** a lo largo del entrenamiento. El conjunto de test **no se toca** hasta la evaluación final, siguiendo las buenas prácticas de machine learning para evitar data leakage en la toma de decisiones de arquitectura.

---

## 10. Evaluación en test y visualización

### Evaluación final en test

```python
test_loader = build_dataloader(test_df)
test_centroids = compute_class_centroids(best_model, train_eval_loader)
test_loss, test_metrics = evaluate_triplet(best_model, test_loader, test_centroids, metrics)
```

La evaluación en test usa los centroides calculados sobre **train** (como en producción). No se re-entrena ni ajusta nada con datos de test.

### Visualización del espacio latente (t-SNE)

```python
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n//2))
embeddings_2d = tsne.fit_transform(embeddings)
plt.scatter(..., color=colors[idx], label=player_name)
```

t-SNE (t-Distributed Stochastic Neighbor Embedding) proyecta los embeddings de 128 dimensiones a 2D preservando la estructura de vecindad local. En el gráfico resultante:
- **Clusters bien separados** → el modelo aprendió representaciones discriminativas.
- **Clusters entremezclados** → el modelo confunde algunos jugadores.

Es una forma visual e intuitiva de evaluar la calidad del espacio latente, más allá de los números de mAP.

### Gráfico de curvas de entrenamiento

Para cada experimento se muestra:
- **Curva de pérdida**: train y val a lo largo de las épocas (detecta overfitting si divergen)
- **Curva de métricas**: mAP, Accuracy y F1-macro en validación

---

## 11. Refinamiento del modelo ganador

Una vez elegido el ganador, se aplican tres estrategias de refinamiento para intentar exprimir el máximo rendimiento:

### R1 — Fine-tuning profundo

```python
model_r1 = copy.deepcopy(best_model)

# Descongelar TODO el backbone
for param in model_r1.backbone.parameters():
    param.requires_grad = True

optimizer_r1 = torch.optim.AdamW([
    {'params': backbone_params_r1, 'lr': 1e-5},   # muy conservador
    {'params': head_params_r1,     'lr': 1e-4},
])
scheduler_r1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_r1, T_max=15)

model_r1, history_r1, best_map_r1 = run_experiment('R1-FineTune', model_r1, ..., n_epochs=15)
```

Parte de los pesos del ganador (no desde cero) y descongelar todo el backbone. Al partir de pesos ya buenos, el LR muy bajo (`1e-5` en backbone) evita destruir lo aprendido mientras ajusta sutilmente las representaciones al dataset específico.

### R2 — Test-Time Augmentation (TTA)

```python
tta_transforms = [
    Compose([Resize, ToTensor, Normalize]),                      # vista original
    Compose([Resize, RandomHorizontalFlip(p=1.0), ...]),         # espejo
    Compose([Resize(246), CenterCrop(224), ...]),                # zoom-in
    Compose([Resize, ColorJitter(0.1, 0.1), ...]),               # variación de color
    Compose([Resize, RandomRotation(8°), ...]),                  # rotación leve
]

# Promedia los logits de las 5 vistas
for tfm in tta_transforms:
    logits_batch = model(augmented_imgs)
    all_logits += logits_batch
all_logits /= len(tta_transforms)
```

**¿Por qué TTA?**  
En vez de hacer una sola predicción por imagen, se predicen 5 versiones aumentadas de la misma imagen y se promedian los logits. Esto reduce la varianza de la predicción: si la versión espejo es más fácil de clasificar que la original, el promedio se beneficia de esa información. **Sin reentrenamiento** → mejora gratis.

### R3 — Ensemble top-2

```python
ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
model1, model2 = ranked[0][1], ranked[1][1]

sims_1 = matmul(embs_from_model1, centroids_1.t())
sims_2 = matmul(embs_from_model2, centroids_2.t())
logits_ensemble = (sims_1 + sims_2) / 2
```

Combina las predicciones de los dos mejores modelos del torneo A/B/C. Cada modelo ve los rostros con diferentes representaciones (FaceNet vs EfficientNet vs MobileNet): donde uno falla, el otro puede acertar. El promedio de similitudes de coseno (en vez de logits directos) es más estable porque ambos modelos generan similitudes normalizadas.

### Comparación y evaluación final

```python
results = {
    'Original': (best_map_val, ...),
    'R1-FineTune': (best_map_r1, ...),
    'R2-TTA': (best_map_r2, ...),
    'R3-Ensemble': (best_map_r3, ...),
}
final_winner = max(results, key=lambda k: results[k][0])
# Evaluar el refinamiento ganador en TEST (una sola vez)
```

El refinamiento ganador (R1, R2 o R3) se compara en validación, y el mejor se evalúa en el conjunto de **test** una única vez.

---

## 12. Compatibilidad con Google Colab

```python
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    DATA_DIR  = '/content/data'
    FACES_DIR = '/content/data/FIFA_2022_ONLY_FACES'
else:
    DATA_DIR  = '../data'
    FACES_DIR = '../data/FIFA_2022_ONLY_FACES'
```

En Colab, si las imágenes de caras no están en `/content/data/`, el notebook las descarga automáticamente desde Google Drive usando el `DRIVE_ZIP_ID`. Para los CSVs, hay dos opciones:
1. **(Recomendada)** Ejecutar primero `02_dataset_preparation.ipynb` en Colab con `SKIP_EXTRACTION=True` — genera las caras y los CSVs en `/content/data/` automáticamente.
2. Si solo se quiere ejecutar el notebook 03, subir un ZIP con los CSVs a Drive y configurar `DRIVE_CSV_ZIP_ID`.

> **Flujo completo en Colab:**  
> Notebook 02 (`SKIP_EXTRACTION=True`) → descarga ZIP de caras desde Drive → genera CSVs → Notebook 03 → descarga caras si no existen → entrena modelos.

---

## Diagrama del flujo completo

```
[Datos]
train.csv + val.csv + test.csv + FIFA_2022_ONLY_FACES/
         ↓
[FaceDataset + DataLoader]  → augmentaciones configurables por experimento
         ↓
[JointEmbeddingNet]  →  backbone (congelado o fine-tuned) + embedding_layer + classifier_head
         ↓
[Entrenamiento]  →  Loss = CrossEntropy + Batch-All Triplet Loss
         ↓
[Evaluación (val)]  →  centroides de train → similitud coseno → mAP / Acc / F1
         ↓
┌──────────────────────────────────────┐
│  Exp A: FaceNet frozen  (25 épocas) │
│  Exp B: EfficientNet partial (30 ép)│
│  Exp C: MobileNet full  (30 épocas) │
└──────────────────────────────────────┘
         ↓ Ganador = max(mAP_val)
[Refinamiento]
  R1: Fine-tune profundo (15 épocas)
  R2: TTA × 5 vistas
  R3: Ensemble top-2
         ↓ Refinamiento ganador = max(mAP_val)
[Evaluación final en TEST]  (una sola vez)
         ↓
[t-SNE del espacio latente]  →  visualización 2D de embeddings
```
