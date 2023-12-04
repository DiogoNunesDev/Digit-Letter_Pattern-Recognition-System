import tensorflow as tf
import numpy as np
import pandas as pd
from show_data import COUNT_DICT, BASE_PATH
import os

"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Setting up the training set

"""

"""
def hexadecimal_to_char(hex):
  integer = int(hex, 16)
  label = chr(integer)
  return label

def labels_dict():
  labels = {}
  for key in COUNT_DICT.keys():
    labels[key] = hexadecimal_to_char(hex= key[-2:])  
  
  return labels

def file_paths_dict():
  file_paths = []
  for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)
    for sub_folder in os.listdir(folder_path):
      if sub_folder.startswith('train_'):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        for file in os.listdir(sub_folder_path):
          file_path = os.path.join(sub_folder_path, file)
          file_paths.append(file_path)
  
  return file_paths

def process_path(file_path, label):

    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    #Normalizing the images
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label
    
label_mappings = labels_dict()
file_paths = file_paths_dict()

def map_file_to_label():
  labels = [label_mappings[os.path.basename(os.path.dirname(path))] for path in file_paths]
  return labels

file_paths = tf.constant(file_paths)
labels = tf.constant(map_file_to_label())

dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

batch_size = 32  #Start => 32
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
"""
"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Model Creation

"""

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 1)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(62, activation='softmax'))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
#tf.keras.utils.plot_model(model, to_file="model.png", show_shapes= True)
    
    