import torch 
from fastapi import FastAPI, HTTPException
from services import get_text_from_url, truncate_text, generate_summary, create_document, get_documents, concatenate_descriptions, generate_answer
from models import InputNewDocument, InputSearch

app = FastAPI()

# Configuración de dispositivo
device = 0 if torch.cuda.is_available() else -1

@app.get("/")
async def read_root():
    return {"message": "¡Bienvenido a tu API con FastAPI!"}


@app.post("/new-document")
async def new_document(data: InputNewDocument):
    
    description = get_text_from_url(data.url)

    if not description:
        raise HTTPException(status_code=400, detail="No se pudo extraer el contenido de la URL")

    description = truncate_text(description)
    result = generate_summary(description)
    keys = result[0].get("summary_text", "").split()

    body = {
        "name": data.name,
        "link": data.url,
        "description": description,
        "keys": keys
    }

    return  {"message": "Creado correctamente"} if create_document(body) else {{"message": "No se pudo crear"}}


@app.post("/search")
async def search(data: InputSearch):
    
    documents = get_documents(data.question)
    context = concatenate_descriptions(documents)
    response = generate_answer(data.question, context)

    return {"answer": response['answer']}
