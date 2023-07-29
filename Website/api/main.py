from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL=tf.keras.models.load_model("../Saved_Model")
CLASS_NAMES=['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']


@app.get("/ping")
async def ping():
   return "Hello, Welcome Back"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file:UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }