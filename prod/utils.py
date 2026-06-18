import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st
from PIL import Image

# ── 1. Definición de la Red ───────────────────────────────────
class JointEmbeddingNet(nn.Module):
    def __init__(self, backbone, embedding_dim=128, num_classes=26):
        super(JointEmbeddingNet, self).__init__()
        self.backbone = backbone
        
        if hasattr(backbone, 'last_linear'):
            # Caso FaceNet (InceptionResnetV1)
            in_features = 512
        elif hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            if isinstance(backbone.classifier, nn.Sequential):
                in_features = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Identity()
            else:
                in_features = backbone.classifier.in_features
                backbone.classifier = nn.Identity()
        else:
            in_features = 512 # Fallback
                
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        logits = self.classifier_head(embeddings)
        return embeddings, logits

# ── 2. Carga Cacheada de Recursos ──────────────────────────────
@st.cache_resource
def load_model_and_centroids(model_path, device_str):
    device = torch.device(device_str)
    
    # Cargar checkpoint unificado
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Reconstruir arquitectura (FaceNet backbone)
    backbone = InceptionResnetV1(pretrained='vggface2')
    model = JointEmbeddingNet(
        backbone, 
        embedding_dim=checkpoint['embedding_dim'], 
        num_classes=checkpoint['num_classes']
    )
    
    # Cargar pesos y configurar en modo evaluación
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Obtener centroides y diccionario de clases
    centroids = checkpoint['centroids'].to(device)
    idx_to_label = checkpoint['idx_to_label']
    
    return model, centroids, idx_to_label

@st.cache_resource
def load_mtcnn(device_str):
    device = torch.device(device_str)
    # keep_all=True para detectar y recortar múltiples rostros en una foto
    return MTCNN(keep_all=True, device=device)

# ── 3. Procesamiento e Inferencia ─────────────────────────────
# Transformación idéntica a la usada durante el entrenamiento
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_faces(image, model, mtcnn, centroids, idx_to_label, threshold, device_str):
    device = torch.device(device_str)
    
    # 1. Detectar cajas de rostros en la imagen original
    boxes, _ = mtcnn.detect(image)
    
    if boxes is None:
        return []
        
    predictions = []
    
    # 2. Iterar sobre cada rostro detectado
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Recortar el rostro usando PIL
        face_crop = image.crop((x1, y1, x2, y2))
        
        # Preprocesar
        face_tensor = preprocess_transform(face_crop).unsqueeze(0).to(device)
        
        # Extraer embedding (con no_grad para mayor velocidad)
        with torch.no_grad():
            embedding, _ = model(face_tensor)
            
        # Calcular similitud coseno (producto punto)
        similarities = torch.matmul(embedding, centroids.t()).squeeze(0)
        max_sim, pred_idx = torch.max(similarities, dim=0)
        
        max_sim = max_sim.item()
        pred_idx = pred_idx.item()
        
        # Determinar si se reconoce o se rechaza según el umbral
        if max_sim >= threshold:
            label = idx_to_label[pred_idx]
            is_recognized = True
        else:
            label = "Ninguno / Desconocido"
            is_recognized = False
            
        predictions.append({
            'box': (x1, y1, x2, y2),
            'face_image': face_crop,
            'label': label,
            'most_similar_player': idx_to_label[pred_idx],
            'confidence': max_sim,
            'is_recognized': is_recognized
        })
        
    return predictions
