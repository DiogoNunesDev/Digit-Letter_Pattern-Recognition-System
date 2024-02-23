from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from src.show_data import BASE_PATH
import os
import matplotlib.pyplot as plt
from src.dataset import create_dataset, compute_weight_classes
"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Setting up the training set

"""

def process_path(file_path, label):

    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    #Normalizing the images
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label

file_paths_np, labels_np = create_dataset('train_')

# Splitting into training and validation sets
train_file_paths, val_file_paths, train_labels, val_labels = train_test_split(
    file_paths_np, labels_np, test_size=0.05, random_state=42, stratify=labels_np
)

print("train test split, DONE")

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_file_paths, val_labels))
print("Tensors created")

# Apply the same preprocessing to both datasets
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

print("Training and Validation sets Complete")

batch_size = 32  #Start => 32

# Shuffling, batching, and prefetching for the training dataset
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Batching and prefetching for the validation dataset
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

print("Training and validation sets, fully prepared!")

"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Model Creation


"""
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 1)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))  
model.add(tf.keras.layers.Dense(62, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
#tf.keras.utils.plot_model(model, to_file="model.png", show_shapes= True)
print("Model successfully created!")


"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Model Training

"""
train_data = {
  '0': 34803, '1': 38049, '2': 34184, '3': 35293, '4': 33432, '5': 31067, '6': 34079, '7': 35796,
  '8': 33884, '9': 33720, 'A': 35050, 'B': 32728, 'C': 33945, 'D': 39560, 'E': 32520, 'F': 40812,
  'G': 33475, 'H': 32710, 'I': 39537, 'J': 39620, 'K': 39568, 'L': 32340, 'M': 40108, 'N': 36596,
  'O': 28680, 'P': 37108, 'Q': 33358, 'R': 32616, 'S': 23827, 'T': 32781, 'U': 28292, 'V': 39608,
  'W': 30156, 'X': 35503, 'Y': 30528, 'Z': 35074, 'a': 33588, 'b': 33306, 'c': 36296, 'd': 34263,
  'e': 28299, 'f': 39888, 'g': 38390, 'h': 38852, 'i': 36244, 'j': 30720, 'k': 33306, 'l': 33874,
  'm': 34242, 'n': 38568, 'o': 35893, 'p': 38416, 'q': 31150, 'r': 31868, 's': 35074, 't': 41586,
  'u': 36881, 'v': 37102, 'w': 35087, 'x': 36660, 'y': 37744, 'z': 35438
}


#Weights: Wj = n_samples / (n_classes * n_samplesj)

class_weights = compute_weight_classes(train_data=train_data)
print("Weights computed!")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, verbose=1, mode="max", restore_best_weights=True)  

reduce_on_plateu = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.2,patience=5,verbose=1,mode="max",min_lr=0.0001,)



num_epochs = 15
history = model.fit(
  train_dataset, 
  epochs=num_epochs, 
  validation_data=val_dataset, 
  callbacks=[early_stopping, reduce_on_plateu],
  class_weight=class_weights
)

print("Training Complete")
model.save('CNN_model.h5')

"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Training results

"""

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
