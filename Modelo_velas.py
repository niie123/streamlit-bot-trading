from transformers import AutoModelForImageClassification
from PIL import Image
import torch
import torchvision.transforms as T

# Cargar modelo
model = AutoModelForImageClassification.from_pretrained("aungkyawwin/Candlestick_Pattern")
model.eval()

# Definir transformaciones manualmente (tamaño 224x224 + normalización como en ViT)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Cargar imagen de ejemplo (ajusta la ruta)
img_path = "recorte_velas.jpg"
img = Image.open(img_path).convert("RGB")
tensor = transform(img).unsqueeze(0)  # Añadir batch dimension

# Clasificar
with torch.no_grad():
    logits = model(tensor).logits
    pred_idx = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=1)[0][pred_idx].item()
    label = model.config.id2label[pred_idx]

print(f"📈 Patrón detectado: {label} ({confidence:.1%} confianza)")
