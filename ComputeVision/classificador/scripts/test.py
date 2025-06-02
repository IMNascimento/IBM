import torch
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Parâmetros
MODEL_PATH = "model/stop_model.pt"
TEST_DIR = Path("data/test")

# Transformação para teste
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Carregar mapeamento real das classes a partir dos dados de treino
train_dataset = datasets.ImageFolder("data/train")
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# Carregar modelo
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Loop de teste
for img_path in TEST_DIR.glob("*.*"):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = idx_to_class[predicted.item()]

    # Mostrar imagem com predição
    plt.imshow(image)
    plt.title(f"{img_path.name} - Predição: {label}")
    plt.axis("off")
    plt.show()
