import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
import cv2

img_sz = (240,215)
batch_sz = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split = 0.2,
    subset = "training",
    seed = 1337,
    image_size = img_sz,
    batch_size = batch_sz,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split = 0.2,
    subset = "validation",
    seed = 1337,
    image_size = img_sz,
    batch_size = batch_sz,
)

class_names = print(train_ds.class_names)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(64,64),
  layers.Rescaling(1./255)
])

num_classes = 7

model = tf.keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
model.summary()
model.save('Hand-Gesture.model')




def prediction(image):
  model = tf.keras.models.load_model('Hand-Gesture.model')
  image = tf.expand_dims(image, 0)
  predictions_arr = model.predict(image)
  pred = np.argmax(predictions_arr)
  return pred



