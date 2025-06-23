from openai import OpenAI
import requests

openai_client = OpenAI()


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
        print('speech to text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text


def text_to_speech(text, voice=""):
    # Configurar a URL da API HTTP do Watson Text-to-Speech
    base_url = '...'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'
    # Adicionando o parâmetro de voz na api_url se o usuário selecionou uma voz preferida
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice
    # Definir os cabeçalhos para nossa requisição HTTP
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }
    # Definir o corpo da nossa requisição HTTP
    json_data = {
        'text': text,
    }
    # Enviar uma requisição HTTP Post para o Serviço Watson Text-to-Speech
    response = requests.post(api_url, headers=headers, json=json_data)
    print('resposta de texto para fala:', response)
    return response.content


def openai_process_message(user_message):
    # Set the prompt for OpenAI Api
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
    # Call the OpenAI Api to process our prompt
    openai_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=4000
    )
    print("openai response:", openai_response)
    # Parse the response to get the response message for our prompt
    response_text = openai_response.choices[0].message.content
    return response_text