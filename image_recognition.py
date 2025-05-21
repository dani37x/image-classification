import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

# Paths and parameters
dataset_path = r"C:\projects\scripts\Mushrooms"
model_path = "mushroom_classifier.h5"
label_map_path = "label_map.json"
img_height, img_width = 150, 150
batch_size = 32
epochs = 30
validation_split = 0.2
seed = 123


def remove_fully_corrupted_images(folder_path):
    num_checked = 0
    num_removed = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.load()  # Forces full read of the image
                num_checked += 1
            except Exception:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                num_removed += 1
    print(
        f"Checked: {num_checked + num_removed}, Removed: {num_removed}, Valid: {num_checked}")


dataset_path = r"C:\projects\scripts\Mushrooms"
remove_fully_corrupted_images(dataset_path)

if __name__ == "__main__":
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Save label map (class index -> class name)
    label_map = {i: class_name for i,
                 class_name in enumerate(train_ds.class_names)}
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)

    # Build model
    model = Sequential([
        # Normalize pixels
        Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(train_ds.class_names), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        # Use sparse if labels are integers (default here)
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save model
    model.save(model_path)

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.title("Accuracy")
    plt.show()
