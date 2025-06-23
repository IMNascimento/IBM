import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Carregar o processador e o modelo pré-treinados
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL da página a ser raspada
url = "https://en.wikipedia.org/wiki/IBM"

# Baixar a página
response = requests.get(url)
# Analisar a página com BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Encontrar todos os elementos img
img_elements = soup.find_all('img')

# Abrir um arquivo para escrever as legendas
with open("captions.txt", "w") as caption_file:
    # Iterar sobre cada elemento img
    for img_element in img_elements:
        img_url = img_element.get('src')

        # Pular se a imagem for um SVG ou muito pequena (provavelmente um ícone)
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # Corrigir a URL se estiver malformada
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue  # Pular URLs que não começam com http:// ou https://

        try:
            # Baixar a imagem
            response = requests.get(img_url)
            # Converter os dados da imagem em uma Imagem PIL
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:  # Pular imagens muito pequenas
                continue

            raw_image = raw_image.convert('RGB')

            # Processar a imagem
            inputs = processor(raw_image, return_tensors="pt")
            # Gerar uma legenda para a imagem
            out = model.generate(**inputs, max_new_tokens=50)
            # Decodificar os tokens gerados para texto
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Escrever a legenda no arquivo, precedida pela URL da imagem
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Erro ao processar a imagem {img_url}: {e}")
            continue