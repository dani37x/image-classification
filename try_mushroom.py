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

    def predict(self, image_path: str, top_k: int = 5) -> List[tuple[str, float]]:
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        return [(self.class_names[i], predictions[i]) for i in top_indices]

    def print_top_predictions(self, image_path: str, top_k: int = 5):
        top_classes = self.predict(image_path, top_k)
        print("Top predicted classes with confidence:")
        for cls, score in top_classes:
            print(f"{cls}: {score:.4f}")


if __name__ == "__main__":
    predictor = MushroomPredictor('models\\mushroom_classifier_v3_EPOCHS_1.h5')
    predictor.print_top_predictions('test_images\\suillus.jpg')
