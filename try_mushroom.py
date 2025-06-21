import sys
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
from typing import List, Optional


class MushroomPredictor:
    DEFAULT_CLASSES = [
        'Agaricus', 'Amanita', 'Boletus', 'Cortinarius',
        'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus'
    ]

    def __init__(self, model_path: str, class_names: Optional[List[str]] = None):
        self.model = load_model(model_path)
        self.class_names = class_names if class_names is not None else self.DEFAULT_CLASSES

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: str, top_k: int = 5) -> List[dict]:
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        return [
            {"class": self.class_names[i],
                "confidence": float(predictions[i]) * 100}
            for i in top_indices
        ]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps(
            {"error": "Missing arguments. Usage: python try_mushroom.py <model_path> <image_path>"}))
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    predictor = MushroomPredictor(model_path)
    results = predictor.predict(image_path)
    print(json.dumps(results))
