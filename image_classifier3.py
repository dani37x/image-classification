import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory


class MushroomClassifier:
    def __init__(self, data_dir, image_size=(224, 224), batch_size=64, epochs=1, model_name="models\\mushroom_classifier"):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.model = None
        self.class_names = []
        self.history = None
        self.train_ds = None
        self.val_ds = None

        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
        ])

    def is_valid_image(self, file_path):
        try:
            img = Image.open(file_path).convert('RGB')
            img.verify()
            return True
        except:
            return False

    def clean_dataset(self):
        print("Cleaning dataset...")
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if not self.is_valid_image(file_path):
                    print(f"Removing corrupted image: {file_path}")
                    os.remove(file_path)

    def load_datasets(self):
        self.train_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=self.image_size,
            batch_size=self.batch_size
        )
        self.val_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.image_size,
            batch_size=self.batch_size
        )
        self.class_names = self.train_ds.class_names

    def preprocess_datasets(self):
        autotune = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.map(lambda x, y: (
            self.data_augmentation(x, training=True), y))
        self.train_ds = self.train_ds.map(lambda x, y: (
            preprocess_input(x), y)).prefetch(buffer_size=autotune)
        self.val_ds = self.val_ds.map(lambda x, y: (
            preprocess_input(x), y)).prefetch(buffer_size=autotune)

    def build_model(self):
        base_model = MobileNetV2(
            input_shape=self.image_size + (3,), include_top=False, weights='imagenet')
        base_model.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        self.model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.history = self.model.fit(
            self.train_ds, validation_data=self.val_ds, epochs=self.epochs)

    def save_model(self):
        filename = f"{self.model_name}_EPOCHS_{self.epochs}.h5"
        self.model.save(filename)
        print(f"Model saved as {filename}")

    def plot_accuracy_loss(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(self.epochs)

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

    def plot_confusion_matrix(self):
        val_images = []
        val_labels = []
        for batch_images, batch_labels in self.val_ds:
            val_images.append(batch_images)
            val_labels.append(batch_labels)

        val_images = tf.concat(val_images, axis=0)
        val_labels = tf.concat(val_labels, axis=0)

        predictions = self.model.predict(val_images)
        predicted_classes = tf.argmax(predictions, axis=1)

        cm = confusion_matrix(val_labels.numpy(), predicted_classes.numpy())
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names)
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_sample_predictions(self, sample_size=9):
        raw_val_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.image_size,
            batch_size=1
        )

        raw_images = list(raw_val_ds.unbatch().take(100))
        random_samples = random.sample(raw_images, sample_size)

        plt.figure(figsize=(10, 10))

        for i, (img, label) in enumerate(random_samples):
            img_array = tf.expand_dims(img, axis=0)
            img_processed = preprocess_input(img_array)

            prediction = self.model.predict(img_processed, verbose=0)
            predicted_label = self.class_names[np.argmax(prediction)]

            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img.numpy().astype("uint8"))
            plt.title(
                f"True: {self.class_names[label]}\nPred: {predicted_label}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def run_all(self):
        self.clean_dataset()
        self.load_datasets()
        self.preprocess_datasets()
        self.build_model()
        self.train_model()
        self.save_model()
        self.plot_accuracy_loss()
        self.plot_confusion_matrix()
        self.plot_sample_predictions()


if __name__ == "__main__":
    classifier = MushroomClassifier(data_dir=r"C:\projects\scripts\Mushrooms")
    classifier.run_all()
