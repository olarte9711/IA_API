import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline, BartTokenizer
from fastapi import HTTPException
import torch

# Configuración de modelo
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


qa_pipelin_answer = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

URL_CREATE_DOCUMENT = "https://proyecto-motor-busqueda.vercel.app/repository"
URL_SEARCH = "https://motor-api-nine.vercel.app/search"

# Función para obtener el texto desde una URL
def get_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])
        text = ' '.join([element.get_text() for element in text_elements])
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    else:
        raise HTTPException(status_code=400, detail="Error al acceder a la URL")

# Función para truncar el texto si es necesario
def truncate_text(text, max_length=1024):
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Función para generar resumen
def generate_summary(text):
    return qa_pipeline(text, max_length=150, min_length=40, do_sample=False)

def generate_answer(question, context):
    return qa_pipelin_answer(question=question, context=context)


def create_document(body):
    try:
        # Hacer la solicitud POST con datos en formato JSON
        response = requests.post(URL_CREATE_DOCUMENT, json=body)
        
        # Imprimir la respuesta completa para depuración
        print(f"Respuesta del servidor: {response.status_code} - {response.text}")

        if response.status_code == 200:
            print("Respuesta del servidor:", response.json())
            return True
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Error al guardar: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error al enviar la solicitud POST: {str(e)}")
        raise HTTPException(status_code=500, detail="Error de red al hacer la solicitud POST")

def get_documents(question):

    body = {
        "question": question,
        "id": 0
    }
    try:
        # Hacer la solicitud POST con datos en formato JSON
        response = requests.post(URL_SEARCH, json=body)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Error al traer documentos: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error al enviar la solicitud POST: {str(e)}")
        raise HTTPException(status_code=500, detail="Error de red al hacer la solicitud POST")

def concatenate_descriptions(data):
    # Inicializamos una variable vacía para almacenar el resultado
    all_descriptions = ""
    
    # Iteramos sobre cada elemento en la lista
    for item in data:
        # Concatenamos la descripción de cada elemento
        all_descriptions += item["description"] + " "
    
    return all_descriptions.strip()