import os
import torch
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate  # Importação atualizada conforme aviso de depreciação
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings  # Novo caminho de importação
from langchain_community.document_loaders import PyPDFLoader  # Novo caminho de importação
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Novo caminho de importação
from langchain_ibm import WatsonxLLM

# Verificar disponibilidade de GPU e definir o dispositivo apropriado para computação.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Variáveis globais
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Função para inicializar o modelo de linguagem e suas embeddings
def init_llm():
    global llm_hub, embeddings

    logger.info("Inicializando WatsonxLLM e embeddings...")

    # Configuração do Modelo Llama
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    PROJECT_ID = "skills-network"

    # Usar os mesmos parâmetros de antes:
    #   MAX_NEW_TOKENS: 256, TEMPERATURE: 0.1
    model_parameters = {
        # "decoding_method": "greedy",
        "max_new_tokens": 256,
        "temperature": 0.1,
    }

    # Inicializar Llama LLM usando a API WatsonxLLM atualizada
    llm_hub = WatsonxLLM(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=model_parameters
    )
    logger.debug("WatsonxLLM inicializado: %s", llm_hub)

    # Inicializar embeddings usando um modelo pré-treinado para representar os dados de texto.
    ### --> se você estiver usando a API huggingFace:
    # Configurar a variável de ambiente para HuggingFace e inicializar o modelo desejado, e carregar o modelo no HuggingFaceHub
    # não se esqueça de remover llm_hub para watsonX

    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "SUA CHAVE DE API"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 600, "max_length": 600})

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings inicializadas com dispositivo do modelo: %s", DEVICE)

# Função para processar um documento PDF
def process_document(document_path):
    global conversation_retrieval_chain

    logger.info("Carregando documento do caminho: %s", document_path)
    # Carregar o documento
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    logger.debug("Carregado %d documento(s)", len(documents))

    # Dividir o documento em partes
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Documento dividido em %d partes de texto", len(texts))

    # Criar um banco de dados de embeddings usando Chroma a partir das partes de texto divididas.
    logger.info("Inicializando loja de vetores Chroma a partir dos documentos...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Loja de vetores Chroma inicializada.")

    # Opcional: Registrar coleções disponíveis se acessíveis (isso pode ser uma API interna)
    try:
        collections = db._client.list_collections()  # _client é interno; ajuste se necessário
        logger.debug("Coleções disponíveis no Chroma: %s", collections)
    except Exception as e:
        logger.warning("Não foi possível recuperar coleções do Chroma: %s", e)

    # Construir a cadeia de QA, que utiliza o LLM e o recuperador para responder perguntas. 
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
        # chain_type_kwargs={"prompt": prompt}  # se você estiver usando um modelo de prompt, descomente esta parte
    )
    logger.info("Cadeia RetrievalQA criada com sucesso.")

# Função para processar um prompt do usuário
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    logger.info("Processando prompt: %s", prompt)
    # Consultar o modelo usando o novo método .invoke()
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    logger.debug("Resposta do modelo: %s", answer)

    # Atualizar o histórico de chat
    chat_history.append((prompt, answer))
    logger.debug("Histórico de chat atualizado. Total de trocas: %d", len(chat_history))

    # Retornar a resposta do modelo
    return answer

# Inicializar o modelo de linguagem
init_llm()
logger.info("Inicialização do LLM e embeddings concluída.")
