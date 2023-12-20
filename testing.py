from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from dataset import create_dataset
import matplotlib.pyplot as plt
import seaborn as sns



model = tf.keras.models.load_model('CNN_model.h5')

def img_to_array(path):
  img = image.load_img(path, target_size=(128, 128), color_mode='grayscale')
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255
  return img_array
    

"""
percentages  = (np.array(prediction) * 100)
np.set_printoptions(precision=2, suppress=True)
print(np.round(percentages, 2))
"""

def get_class(predictions):  
  char_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  #class_indices = np.argmax(predictions, axis=0)
  return [char_labels[index] for index in predictions]
  
def image_generator(paths):
    for i in range(0, len(paths), 32):
        batch_paths = paths[i:i + 32]
        batch_images = np.vstack([img_to_array(path) for path in batch_paths])
        yield batch_images
        
        
X_test_paths, y_test = create_dataset('hsf_')



def predict(X_test_paths):
  gen = image_generator(X_test_paths)
  y_pred = []
  for _ in range(len(X_test_paths) // 32):
      batch_images = next(gen)
      batch_pred = model.predict(batch_images)
      y_pred.extend(batch_pred)
  
  remaining = len(X_test_paths) % 32
  if remaining:
    batch_images = next(gen)
    batch_pred = model.predict(batch_images[:remaining])
    y_pred.extend(batch_pred)
  
  return y_pred

y_pred = predict(X_test_paths)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
y_pred_classes = get_class(y_pred)
y_test_classes = get_class(y_test)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print("Accuracy:", accuracy)
print(classification_report(y_test_classes, y_pred_classes))

# Plotting confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(60, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualization of predictions
plt.figure(figsize=(60, 5))
plt.plot(y_pred_classes, label='Predicted')
plt.plot(y_test_classes, label='Actual')
plt.ylabel("Class")
plt.legend()
plt.show()




    
    
