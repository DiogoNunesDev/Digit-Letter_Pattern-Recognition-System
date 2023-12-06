from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from show_data import COUNT_DICT, BASE_PATH
import os
import matplotlib.pyplot as plt
"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Setting up the training set

"""
def hexadecimal_to_char(hex):
  integer = int(hex, 16)
  label = chr(integer)
  return label

def labels_dict():
  char_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  return {label: i for i, label in enumerate(char_labels)}

print("Labels dict complete")

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
  
  print("File paths dict complete")
  return file_paths

def process_path(file_path, label):

    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    #Normalizing the images
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label
    
label_mappings = labels_dict()
file_paths = file_paths_dict()

def map_file_to_label(file_paths, labels_map):
    labels = []
    for path in file_paths:
        label_key = path.split('\\')[-3] 
        label_char = hexadecimal_to_char(hex=label_key)
        label = label_mappings[label_char]
        labels.append(label)
    return labels

labels = map_file_to_label(file_paths, label_mappings)

labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=62)

print("One hot completed")

# Convert to numpy arrays for compatibility with train_test_split
file_paths_np = np.array(file_paths) 
labels_np = np.array(labels_one_hot)
  
# Splitting into training and validation sets
train_file_paths, val_file_paths, train_labels, val_labels = train_test_split(
    file_paths_np, labels_np, test_size=0.02, random_state=42, stratify=labels_np
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

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(62, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#print(model.summary())
#tf.keras.utils.plot_model(model, to_file="model.png", show_shapes= True)
print("Model successfully created!")


"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Model Training

"""
  
early_stopping_callblack = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)  
  
num_epochs = 10
history = model.fit(
  train_dataset, 
  epochs=num_epochs, 
  validation_data=val_dataset, 
  callbacks=[early_stopping_callblack]
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
