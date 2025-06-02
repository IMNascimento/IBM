import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Parâmetros
DATA_DIR = "data/train"
MODEL_PATH = "model/stop_model.pt"
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.001

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo pré-treinado
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: stop / not_stop

# Treino
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🖥️ Usando GPU: {torch.cuda.get_device_name(device)}")
else:
    print("⚠️ Usando CPU")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("🔁 Iniciando treinamento...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"📦 Época {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# Salvar modelo
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Modelo salvo em: {MODEL_PATH}")
