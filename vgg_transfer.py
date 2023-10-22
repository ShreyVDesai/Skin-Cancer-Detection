import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16

# Load data
data = pd.read_csv("HAM10000_metadata.csv")

# Load images
# Load images
images = []
for i in range(len(data)):
    img_path = "HAM10000_images/" + data["image_id"][i] + ".jpg"
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img.astype('float32') / 255.0  # normalize pixel values
    images.append(img)


# Create X and y
X = np.array(images)
y = np.array(data["dx"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data generator
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

# Create base model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(7, activation="softmax")(x)

# Create model
model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])

# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=8),
                    validation_data=(X_test, y_test),
                    epochs=10,
                    steps_per_epoch=len(X_train) / 32)

# Plot accuracy and loss
sns.set_style("darkgrid")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Loss")
plt.legend()
plt.show()
