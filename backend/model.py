from pydantic import BaseModel

class Todo(BaseModel):
    title: str
    description: str

class Image(BaseModel):
    imageSrc: str

class Input(BaseModel):
    base64str : str
    threshold : float