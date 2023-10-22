import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# load the metadata
metadata = pd.read_csv('HAM10000_metadata.csv')

# create a dictionary to map the labels to integers
label_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

# create a list of dictionaries that contains the image paths and labels
data = []
for i in range(len(metadata)):
    img_path = 'HAM10000_images/' + metadata['image_id'][i] + '.jpg'
    label = label_dict[metadata['dx'][i]]
    data.append({'image_id': metadata['image_id'][i], 'label': label, 'path': img_path})

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# define the batch size and new size of the images
batch_size = 32
new_size = (224, 224)

# define a function that generates batches of data
def data_generator(data, batch_size, new_size):
    while True:
        # shuffle the data
        indices = np.random.permutation(len(data))
        data = np.array(data)[indices]
        
        # generate batches of data
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_images = []
            batch_labels = []
            for j in range(len(batch_data)):
                img = Image.open(batch_data[j]['path'])
                img = img.resize(new_size)
                img = np.array(img) / 255.0
                label = batch_data[j]['label']
                batch_images.append(img)
                batch_labels.append(label)
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            batch_labels = to_categorical(batch_labels, num_classes=7)
            yield batch_images, batch_labels

# create the data generators
train_generator = data_generator(train_data, batch_size, new_size)
test_generator = data_generator(test_data, batch_size, new_size)

# define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model using the data generators
model.fit(train_generator, epochs=15, steps_per_epoch=len(train_data) // batch_size,
          validation_data=test_generator, validation_steps=len(test_data) // batch_size)

# evaluate the model on the test data using the test generator
test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_data) // batch_size)
print('Test accuracy:', test_acc)
