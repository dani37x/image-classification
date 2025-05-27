import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
DATA_DIR = r"C:\projects\scripts\Mushrooms"

# Filter and clean images


def is_valid_image(file_path):
    try:
        img = Image.open(file_path).convert('RGB')
        img.verify()
        return True
    except:
        return False


def clean_directory(directory):
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if not is_valid_image(file_path):
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)


print("Cleaning dataset...")
clean_directory(DATA_DIR)

# Load dataset
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (
    preprocess_input(x), y)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)
                    ).prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Build model
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                         include_top=False, weights='imagenet')
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
EPOCHS = 15
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save model
model.save(f'mushroom_classifier2 EPOCHS:{EPOCHS}.h5')
print("Model saved as mushroom_classifier.h5")

# Predict and show examples correctly (no preprocessing on displayed images)
val_ds_unprocessed = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=9  # just a few images for visualization
)

class_names = val_ds_unprocessed.class_names

plt.figure(figsize=(12, 8))
for images, labels in val_ds_unprocessed.take(1):
    # Preprocess only for prediction
    processed_images = preprocess_input(images)
    predictions = model.predict(processed_images)

    for i in range(len(images)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # original image
        predicted_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[labels[i]]
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis("off")

plt.tight_layout()
plt.show()
