import torch
from transformers import pipeline
import gradio as gr

# Função para transcrever áudio usando o modelo OpenAI Whisper
def transcript_audio(audio_file):
    # Inicializa o pipeline de reconhecimento de fala
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # Transcreve o arquivo de áudio e retorna o resultado
    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Configura a interface Gradio
audio_input = gr.Audio(sources="upload", type="filepath")  # Entrada de áudio
output_text = gr.Textbox()  # Saída de texto

# Cria a interface Gradio com a função, entradas e saídas
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Aplicativo de Transcrição de Áudio",
                     description="Envie o arquivo de áudio")

# Inicia o aplicativo Gradio
iface.launch(server_name="0.0.0.0", server_port=7860)