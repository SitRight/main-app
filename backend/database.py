import motor.motor_asyncio
from model import Todo
from model import Image

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017/')
database = client.TodoList
user_collection = database.todo
image_collection = database.image
prediction_collection = database.prediction

async def fetch_one_todo(title):
    document = await user_collection.find_one({"title": title})
    return document

async def fetch_all_todos():
    todos = []
    cursor = user_collection.find({})
    async for document in cursor:
        todos.append(Todo(**document))
    return todos

async def create_todo(todo):
    document = todo
    result = await user_collection.insert_one(document)
    return document

async def create_image(image):
    document = image
    result = await image_collection.insert_one(document)
    return document

async def fetch_all_images():
    images = []
    cursor = image_collection.find({})
    async for document in cursor:
        images.append(Image(**document))
    return images


async def create_prediction(prediction):
    document = prediction
    result = await prediction_collection.insert_one(document)
    return document