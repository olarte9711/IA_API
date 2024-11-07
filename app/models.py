from pydantic import BaseModel

class InputNewDocument(BaseModel):
    name: str
    url: str

class InputSearch(BaseModel):
    question: str
    id: int