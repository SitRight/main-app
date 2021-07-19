from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import base64
import io
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions


from model import Todo
from model import Image
from model import Input
from database import (
    fetch_one_todo,
    fetch_all_todos,
    create_todo,
    create_image,
    fetch_all_images,
    create_prediction
)

# an HTTP-specific exception class  to generate exception information

app = FastAPI()

origins = [
    "http://localhost:3000",
]

# what is a middleware? 
# software that acts as a bridge between an operating system or database and applications, especially on a network.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

def predict(image: Image.Image):
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    result = decode_predictions(model.predict(image), 2)[0]
    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)
    return response

model = load_model()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/api/todo")
async def get_todo():
    response = await fetch_all_todos()
    return response

@app.get("/api/todo/{title}", response_model=Todo)
async def get_todo_by_title(title):
    response = await fetch_one_todo(title)
    if response:
        return response
    raise HTTPException(404, f"There is no todo with the title {title}")

@app.post("/api/todo/", response_model=Todo)
async def post_todo(todo: Todo):
    response = await create_todo(todo.dict())
    if response:
        return response
    raise HTTPException(400, "Something went wrong")

@app.get("/api/image")
async def get_image():
    response = await fetch_all_images()
    return response

@app.post("/api/image/", response_model=Image)
async def post_image(image: Image):
    response = await create_image(image.dict())
    if response:
        return response
    raise HTTPException(400, "Something went wrong")

@app.post("/predict/image", response_model=Input)
async def predict_api(base64str):
    img = await base64str_to_PILImage(base64str)
    prediction = predict(img)
    response = await create_prediction(prediction.dict())
    if response:
        return response
    raise HTTPException(400, "Something went wrong")
    



