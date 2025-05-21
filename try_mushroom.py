
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json


# Config
model_path = "mushroom_classifier.h5"
label_map_path = "label_map.json"
img_height, img_width = 150, 150

# Load model and labels
model = tf.keras.models.load_model(model_path)

with open(label_map_path, 'r') as f:
    label_map = json.load(f)


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]  # single prediction
    class_index = np.argmax(prediction)
    class_label = label_map[str(class_index)]
    confidence = float(prediction[class_index])

    print(f"Predicted class: {class_label}, Confidence: {confidence:.4f}")

    return class_label, confidence


# Example usage
if __name__ == "__main__":
    test_img = r"C:\projects\scripts\agaricus.jpg"
    predict_image(test_img)
