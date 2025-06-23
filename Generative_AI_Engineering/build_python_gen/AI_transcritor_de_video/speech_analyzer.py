import torch
import os
import gradio as gr

#from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

meus_credenciais = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}
params = {
        GenParams.MAX_NEW_TOKENS: 800, # O número máximo de tokens que o modelo pode gerar em uma única execução.
        GenParams.TEMPERATURE: 0.1,   # Um parâmetro que controla a aleatoriedade da geração de tokens. Um valor mais baixo torna a geração mais determinística, enquanto um valor mais alto introduz mais aleatoriedade.
    }

modelo_LLAMA2 = Model(
        model_id= 'meta-llama/llama-3-2-11b-vision-instruct', 
        credentials=meus_credenciais,
        params=params,
        project_id="skills-network",  
        )

llm = WatsonxLLM(modelo_LLAMA2)  

#######------------- Modelo de Prompt-------------####

temp = """
<s><<SYS>>
Liste os pontos principais com detalhes do contexto: 
[INST] O contexto : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_para_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- Fala para Texto-------------####

def transcrever_audio(arquivo_audio):
    # Inicializa o pipeline de reconhecimento de fala
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # Transcreve o arquivo de áudio e retorna o resultado
    texto_transcrito = pipe(arquivo_audio, batch_size=8)["text"]
    resultado = prompt_para_LLAMA2.run(texto_transcrito)

    return resultado

#######------------- Gradio-------------####

entrada_audio = gr.Audio(sources="upload", type="filepath")
texto_saida = gr.Textbox()

iface = gr.Interface(fn= transcrever_audio, 
                    inputs= entrada_audio, outputs= texto_saida, 
                    title= "Aplicativo de Transcrição de Áudio",
                    description= "Faça o upload do arquivo de áudio")

iface.launch(server_name="0.0.0.0", server_port=7860)