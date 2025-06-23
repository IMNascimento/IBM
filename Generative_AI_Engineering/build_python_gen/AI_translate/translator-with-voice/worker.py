# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
import requests

# placeholder for Watsonx_API and Project_id incase you need to use the code outside this environment
# API_KEY = "Your WatsonX API"
PROJECT_ID= "skills-network"

# Define the credentials 
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    #"apikey": API_KEY
}
    
# Specify model_id that will be used for inferencing
model_id = ModelTypes.FLAN_UL2

# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)


def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP Api url
    base_url = '...'
    api_url = base_url+'/speech-to-text/api/v1/recognize'
    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }
    # Set up the body of our HTTP request
    body = audio_binary
    # Send a HTTP Post request
    response = requests.post(api_url, params=params, data=audio_binary).json()
    # Parse the response to get our transcribed text
    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text

def text_to_speech(text, voice=""):
    # Configurar a URL da API HTTP da Watson Text-to-Speech
    base_url = '...'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'
    # Adicionando o parâmetro de voz na api_url se o usuário selecionou uma voz preferida
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice
    # Configurar os cabeçalhos para nossa solicitação HTTP
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }
    # Configurar o corpo da nossa solicitação HTTP
    json_data = {
        'text': text,
    }
    # Enviar uma solicitação HTTP Post para o Serviço Watson Text-to-Speech
    response = requests.post(api_url, headers=headers, json=json_data)
    print('Resposta do Texto para Fala:', response)
    return response.content

def watsonx_process_message(user_message):
    # Defina o prompt para a API Watsonx
    prompt = f"""Responda à consulta: ```{user_message}```"""
    response_text = model.generate_text(prompt=prompt)
    print("resposta do watsonx:", response_text)
    return response_text
