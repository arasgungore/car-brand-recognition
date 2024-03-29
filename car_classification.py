# -*- coding: utf-8 -*-
"""Car_Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yss-MRyIfZv-PVNuuVgqTELTBY_zxaXm
"""

from google.colab import drive
import os
import shutil

drive.flush_and_unmount()

# Define the path to the Colab workspace
workspace_path = '/content/'

shutil.rmtree(workspace_path, ignore_errors=True)

# Create the destination directory if it doesn't exist
os.makedirs(workspace_path, exist_ok=True)

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the zip file in Google Drive
zip_file_path = '/content/drive/MyDrive/Datasets/stanford_car_dataset_by_classes.zip'

# Copy the zip file from Google Drive to the Colab workspace
shutil.copy2(zip_file_path, workspace_path)

# Change the working directory to the destination path
os.chdir(workspace_path)

# Unzip the file
shutil.unpack_archive(os.path.basename(zip_file_path), workspace_path)

# Remove zip file
os.remove(os.path.basename(zip_file_path))

# Unmount Google Drive
drive.flush_and_unmount()

# Define the paths for the training and test datasets
TRAIN_DIR = "/content/stanford_car_dataset_by_classes/car_data/car_data/train"
TEST_DIR = "/content/stanford_car_dataset_by_classes/car_data/car_data/test"
COMBINED_DIR = "/content/stanford_car_dataset_by_classes/car_data/car_data/combined"

# Create the combined dataset directory
os.makedirs(COMBINED_DIR, exist_ok=True)

# Move the training dataset subdirectories to the combined dataset directory
for class_dir in os.listdir(TRAIN_DIR):
    source_dir = os.path.join(TRAIN_DIR, class_dir)
    destination_dir = os.path.join(COMBINED_DIR, class_dir)
    shutil.copytree(source_dir, destination_dir)

# Copy the test dataset subdirectories to the combined dataset directory
for class_dir in os.listdir(TEST_DIR):
    source_dir = os.path.join(TEST_DIR, class_dir)
    destination_dir = os.path.join(COMBINED_DIR, class_dir)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np

# tf.config.gpu_options.allow_growth = True
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

try:
    session
except NameError:
    session = None

if session is not None:
    session.close()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMG_SIZE = 256
BATCH_SIZE = 16

def count_no_of_files(directory):
    count = sum([len(files) for r, d, files in os.walk(directory)])
    print('File count in "' + directory + '":', count)

count_no_of_files(TRAIN_DIR)
count_no_of_files(TEST_DIR)
count_no_of_files(COMBINED_DIR)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range = 0.1,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    validation_split=0.1)

train_generator = datagen.flow_from_directory(
    directory = COMBINED_DIR, 
    class_mode = "categorical", 
    seed = 1,
    target_size = (IMG_SIZE,IMG_SIZE), 
    subset='training',
    batch_size = BATCH_SIZE)

valid_generator = datagen.flow_from_directory(
    directory = COMBINED_DIR, 
    seed = 1,
    class_mode = "categorical", 
    target_size = (IMG_SIZE,IMG_SIZE), 
    subset='validation',
    batch_size = BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# # Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

base_model.summary()

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(196, activation = 'softmax')
])

model.summary()

base_learning_rate = 0.0001
model.compile(Adam(learning_rate = base_learning_rate),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_generator,
                    validation_data=valid_generator,
                    epochs=10)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,10.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""## Fine Tune the Model ##"""

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 60

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

IMG_SIZE = 256
BATCH_SIZE = 8


datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range = 0.1,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    validation_split=0.1)

train_generator = datagen.flow_from_directory(
    directory = COMBINED_DIR, 
    class_mode = "categorical", 
    seed = 1,
    target_size = (IMG_SIZE,IMG_SIZE), 
    subset='training',
    batch_size = BATCH_SIZE)

valid_generator = datagen.flow_from_directory(
    directory = COMBINED_DIR, 
    seed = 1,
    class_mode = "categorical", 
    target_size = (IMG_SIZE,IMG_SIZE), 
    subset='validation',
    batch_size = BATCH_SIZE)

history = model.fit(train_generator,
                    validation_data=valid_generator,
                    epochs=5)

train_generator.class_indices

drive.mount('/content/drive')
model_path = "/content/drive/My Drive/Model/car_classification_ResNet50_Preprocessed_fineTuned_allData.h5"
model.save(model_path)
drive.flush_and_unmount()
model_path = "/content/Model/car_classification_ResNet50_Preprocessed_fineTuned_allData.h5"
model.save(model_path)
del(model)

from tensorflow.keras.models import load_model
from google.colab import drive

drive.mount('/content/drive')
model_path = "/content/drive/My Drive/Model/car_classification_ResNet50_Preprocessed_fineTuned_allData.h5"
model = load_model(model_path)
drive.flush_and_unmount()

"""## Checking Model Sanity on Training Data"""

import cv2
path = TRAIN_DIR + "/Volkswagen Beetle Hatchback 2012/00019.jpg"
img = cv2.imread(path)
image_x = 256
image_y = 256
img = cv2.resize(img, (image_x, image_y))
img = np.array(img, dtype=np.float32)
img = np.reshape(img, (-1, image_x, image_y, 3))

image = mpimg.imread(path)
plt.imshow(image)
plt.show()

pred = model.predict(img)
print(np.argmax(pred))
# print(class_list[np.argmax(pred)])
car_label = list(filter(lambda x: train_generator.class_indices[x] == np.argmax(pred), train_generator.class_indices))[0]
print(car_label)
# print(labels[np.argmax(pred)])
# train_generator.class_indices

from sklearn.metrics import classification_report, confusion_matrix

datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range = 0.1,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    validation_split=0.9999999)

test_generator = datagen.flow_from_directory(
    directory = TEST_DIR, 
    seed = 1,
    class_mode = "categorical", 
    target_size = (IMG_SIZE,IMG_SIZE), 
    subset='validation',
    batch_size = BATCH_SIZE)

# Make predictions on the test set
predictions = model.predict(test_generator)

# Get the class with the highest probability for each sample
y_pred = np.argmax(predictions, axis=1)

# Get the true classes
y_true = test_generator.classes

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate precision, recall, and F1 score
report = classification_report(y_true, y_pred, target_names=class_names)

print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)