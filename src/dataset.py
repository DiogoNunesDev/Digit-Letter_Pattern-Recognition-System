import tensorflow as tf
import numpy as np
import pandas as pd
from show_data import BASE_PATH
import os
import matplotlib.pyplot as plt
"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Setting up the datasets 

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

#type: 'train_' or 'hsf_'  
def file_paths_dict(type): 
  file_paths = []
  for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)
    for sub_folder in os.listdir(folder_path):
      if sub_folder.startswith(type):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        for file in os.listdir(sub_folder_path):
          file_path = os.path.join(sub_folder_path, file)
          file_paths.append(file_path)
  
  print("File paths dict complete")
  return file_paths

def map_file_to_label(file_paths, label_mappings):
    labels = []
    for path in file_paths:
        label_key = path.split('\\')[-3] 
        label_char = hexadecimal_to_char(hex=label_key)
        label = label_mappings[label_char]
        labels.append(label)
    return labels

def create_dataset(type):
  label_mappings = labels_dict()
  print("Labels dict complete")
  
  file_paths = file_paths_dict(type=type)
  print("File paths complete")
  
  labels = map_file_to_label(file_paths, label_mappings)
  labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=62)
  print("One hot completed")
  
  # Convert to numpy arrays for compatibility with train_test_split
  file_paths_np = np.array(file_paths) 
  labels_np = np.array(labels_one_hot)
  print("Convertion to np array complete")
  return file_paths_np, labels_np

def compute_weight_classes(train_data):
    label_mappings = labels_dict()  # Get the label to integer index mapping
    total_samples = sum(train_data.values()) * 0.095
    num_classes = 62  # Total number of classes

    # Modify the class weights calculation to use integer indices
    class_weights = {label_mappings[label]: total_samples / (num_classes * num_samples) 
                     for label, num_samples in train_data.items()}

    return class_weights

