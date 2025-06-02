import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import numpy as np

# ===== Constants =====
IPOS = "mushroom_classifier_v3"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
DATA_DIR = r"C:\projects\scripts\Mushrooms"
EPOCHS = 15

# ===== Data Augmentation =====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])

# ===== Clean dataset =====


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

# ===== Load dataset =====
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

# ===== Preprocess datasets =====
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (
    preprocess_input(x), y)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)
                    ).prefetch(buffer_size=AUTOTUNE)

# ===== Build model =====
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                         include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===== Train model =====

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ===== Save model =====
model_name = f"{IPOS}_EPOCHS_{EPOCHS}.h5"
model.save(model_name)
print(f"Model saved as {model_name}")

# ===== PLOT 1: Accuracy & Loss =====
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.tight_layout()
plt.show()

# ===== PLOT 2: Confusion Matrix =====
val_images = []
val_labels = []

for batch_images, batch_labels in val_ds:
    val_images.append(batch_images)
    val_labels.append(batch_labels)

val_images = tf.concat(val_images, axis=0)
val_labels = tf.concat(val_labels, axis=0)

predictions = model.predict(val_images)
predicted_classes = tf.argmax(predictions, axis=1)

cm = confusion_matrix(val_labels.numpy(), predicted_classes.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== PLOT 3: Raw Images & Predictions =====
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
