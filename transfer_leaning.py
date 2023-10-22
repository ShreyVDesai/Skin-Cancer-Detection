import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('HAM10000_metadata.csv')
labels = df['dx']
image_path = df['image_id'].apply(lambda x: x + '.jpg')

# Split the data into training and validation sets
train_path, val_path, train_labels, val_labels = train_test_split(image_path, labels, test_size=0.2, random_state=42)

# Load the pre-trained model
base_model = keras.applications.ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification layer
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(7, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing
def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = keras.applications.resnet50.preprocess_input(img_array)
    return img_array

train_images = np.array([preprocess_image('HAM10000_images/' + img_path) for img_path in train_path])
val_images = np.array([preprocess_image('HAM10000_images_part_1/' + img_path) for img_path in val_path])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
