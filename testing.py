import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import numpy as np



model = tf.keras.models.load_model('CNN_model.h5')

def img_to_array(path):
  img = image.load_img(path, target_size=(128, 128), color_mode='grayscale')
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255
  return img_array
    
prediction = model.predict(img_to_array("Inputs\\train_4a_00029.png"))

"""
percentages  = (np.array(prediction) * 100)
np.set_printoptions(precision=2, suppress=True)
print(np.round(percentages, 2))
"""

def get_class(predicion):
  index = np.argmax(prediction, axis=1)[0]
  print(index)
  char_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  
  return char_labels[index]
  
  
print(get_class(prediction))
        
  
      
    
    
