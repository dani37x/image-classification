import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import random

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
DATA_DIR = r"C:\projects\scripts\Mushrooms"

# Remove corrupted images


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
AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation for training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (
    preprocess_input(x), y)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)
                    ).prefetch(buffer_size=AUTOTUNE)

# Compute class weights
y_train = []
for _, labels in train_ds.unbatch():
    y_train.append(labels.numpy())

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Build model
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                         include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
    # !

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model and store history
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Save model
model_path = f'mushroom_classifier_{EPOCHS}epochs.h5'
model.save(model_path)
print(f"Model saved as {model_path}")

# Plot training & validation accuracy/loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion matrix
y_true = []
y_pred = []

for images, labels in val_ds.unbatch():
    pred = model.predict(tf.expand_dims(images, axis=0), verbose=0)
    y_pred.append(np.argmax(pred))
    y_true.append(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Reload raw validation images (without preprocessing) for visualization
raw_val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=1
)

# Random 9 sample predictions
raw_images = list(raw_val_ds.unbatch().take(100))
random_samples = random.sample(raw_images, 9)

plt.figure(figsize=(10, 10))

for i, (img, label) in enumerate(random_samples):
    img_array = tf.expand_dims(img, axis=0)
    img_processed = preprocess_input(img_array)

    prediction = model.predict(img_processed, verbose=0)
    predicted_label = class_names[np.argmax(prediction)]

    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"True: {class_names[label]}\nPred: {predicted_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
