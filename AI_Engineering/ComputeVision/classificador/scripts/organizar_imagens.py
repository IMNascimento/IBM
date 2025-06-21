import os
import shutil
from pathlib import Path
import random

# Diretórios
RAW_STOP = Path("raw/stop")
RAW_NOT_STOP = Path("raw/not_stop")

TRAIN_STOP = Path("data/train/stop")
TRAIN_NOT_STOP = Path("data/train/not_stop")
TEST_DIR = Path("data/test")

# Criar pastas se não existirem
for folder in [TRAIN_STOP, TRAIN_NOT_STOP, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Coletar imagens
stop_images = sorted(list(RAW_STOP.glob("*.*")))
not_stop_images = sorted(list(RAW_NOT_STOP.glob("*.*")))

random.shuffle(stop_images)
random.shuffle(not_stop_images)

def copiar_imagens(imagens, destino_treino, label):
    split_idx = int(0.8 * len(imagens))
    treino = imagens[:split_idx]
    teste = imagens[split_idx:]

    # Treino
    for img in treino:
        shutil.copy(img, destino_treino / img.name)

    # Teste
    for i, img in enumerate(teste):
        ext = img.suffix.lower()
        nome_seguro = f"{label}_{i+1}{ext}"
        shutil.copy(img, TEST_DIR / nome_seguro)

# Executar cópia
copiar_imagens(stop_images, TRAIN_STOP, "stop")
copiar_imagens(not_stop_images, TRAIN_NOT_STOP, "not_stop")

print("✅ Organização finalizada com sucesso!")
