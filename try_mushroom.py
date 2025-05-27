from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Load model
model = load_model('mushroom_classifier.h5')

# Define class names in English (order must match the training folder order)
class_names = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius',
               'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

# Load and preprocess image
image_path = r'lactarius.jpg'
img = Image.open(image_path).convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)[0]

# Top 5 predictions
top_indices = predictions.argsort()[-5:][::-1]
top_classes = [(class_names[i], predictions[i]) for i in top_indices]

# Print
print("Top 5 predicted classes with confidence:")
for cls, score in top_classes:
    print(f"{cls}: {score:.4f}")
