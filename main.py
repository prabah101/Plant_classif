from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from enum import Enum


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POTATO_MODEL = tf.keras.models.load_model("./saved_models/2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

class AvailablePlants(str, Enum):
    tomato = "tomato"
    potato = "potato"
    bell_pepper = "bell_pepper"

@app.get("/hello")
async def hello():
    return "This is a Plant Disease classification API"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/{plant}")
async def predict(
    file: UploadFile = File(...),
    plant: AvailablePlants = AvailablePlants.potato
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = POTATO_MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)


    return {
        'plant': plant,
        'class': predicted_class,
        'confidence': float(confidence)
    }


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
