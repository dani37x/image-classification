from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io

app = FastAPI()


class MushroomPredictor:
    DEFAULT_CLASSES = [
        'Agaricus', 'Amanita', 'Boletus', 'Cortinarius',
        'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus'
    ]

    def __init__(self, model_path: str, class_names: List[str] = None):
        self.model = load_model(model_path)
        self.class_names = class_names if class_names is not None else self.DEFAULT_CLASSES

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[dict]:
        img_array = self.preprocess_image(image_bytes)
        predictions = self.model.predict(img_array)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        return [
            {"class": self.class_names[i],
                "confidence": float(predictions[i]) * 100}
            for i in top_indices
        ]


# Initialize your predictor here with your model path:
predictor = MushroomPredictor(
    "models\\mushroom_classifier_EPOCHS_1_EPOCHS_1.h5")


@app.post("/predict/")
async def predict(image: UploadFile = File(...), top_k: int = 5):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, detail="Invalid image type. Use JPEG or PNG.")

    image_bytes = await image.read()
    predictions = predictor.predict(image_bytes, top_k=top_k)
    return {"predictions": predictions}
