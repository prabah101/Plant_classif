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
POTATO_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

BELL_PEPPER_MODEL = tf.keras.models.load_model("./saved_models/3")
BELL_PEPPER_CLASS_NAMES = ["Bacterial Spot", "Healthy"]

class AvailablePlants(str, Enum):
    tomato = "tomato"
    potato = "potato"
    bell_pepper = "bell_pepper"

@app.get("/")
async def hello():
    return "This is a Plant Disease classification API. Visit '/docs' for more info."

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
    predicted_class = POTATO_CLASS_NAMES[np.argmax(predictions[0])]
    
    if(plant == "potato"):
        predictions = POTATO_MODEL.predict(img_batch)
        predicted_class = POTATO_CLASS_NAMES[np.argmax(predictions[0])]
    elif(plant == "bell_pepper"):
        predictions = BELL_PEPPER_MODEL.predict(img_batch)
        predicted_class = BELL_PEPPER_CLASS_NAMES[np.argmax(predictions[0])]

    confidence = round(100 * (np.max(predictions[0])), 2)


    return {
        'plant': plant,
        'class': predicted_class,
        'confidence': float(confidence)
    }


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
