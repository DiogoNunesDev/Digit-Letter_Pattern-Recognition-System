import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import uuid
import cv2
from show_data import COUNT_DICT

"""

THIS FILE IS RESPONSABLE FOR EVERYTHING CONCERNING DATA PREPROCESSING

"""

BASE_PATH = "C:\\Users\\diogo\\OneDrive\\Ambiente de Trabalho\\Projects\\Sistema de Reconhecimento de Padrões\\byclass"

#Auxiliar functions for the start of the dataset analysis

def clean_mit_data(base_path):

  for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
      # iterate over the Character folders
    for content in os.listdir(folder_path):
    #If it is a .mit file, remove it
      content_path = os.path.join(folder_path, content)
     #if it is a folder, do nothing
      if os.path.isdir(content_path):
        continue
      if content.lower().endswith('.mit'):
        os.remove(content_path)

def check_corruption(base_path):
  for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    # iterate over the Character folders
    for sub_folder in os.listdir(folder_path):
      sub_folder_path = os.path.join(folder_path, sub_folder)
      for file in os.listdir(sub_folder_path):
        if file.lower().endswith('.png'):
          image_path = os.path.join(sub_folder_path, file)
          try:
            with Image.open(image_path) as img:
              img.verify()
          except Exception as e:
            #If the system cannot open the image, then it is deleted. One time use! 
            print(f"Corrupted Image: {image_path}")
            os.remove(image_path)




#Inputed data transformation section

def preprocess_inputed_image(input_image_path):
  """
  This function is used everytime we receive a new image as an input. It will transform it in a way that it looks like the ones in the dataset. 
  """

  original_image = Image.open(input_image_path)
  
  #Convert the image to a NumPy array, to access each pixel and change it to a greyscale
  image_array = np.array(original_image)

  height, width, _ = image_array.shape
  
  #Make the image grayscale and use contrast
  threshold = 100
  for i in range(height):
    for j in range(width):
      pixel = image_array[i, j]
      red = pixel[0]
      green = pixel[1]
      blue = pixel[2]
      #Apply this formula to make it greyscale: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
      grayscale_value = int(0.2226 * red + 0.7152 * green + 0.0722 * blue)
      grayscale_value = max(0, min(255, grayscale_value))
      # make the pixel either perfect black or perfect white
      if grayscale_value <= threshold:
        image_array[i,j] = [0,0,0]
      else:
        image_array[i,j] = [255,255,255]
  #Convert the grayscale array back to an image format
  gray_image = Image.fromarray(image_array)

  # Resize the image to a 128x128
  resized_image = gray_image.resize((128, 128), Image.LANCZOS)

  resized_image_array = np.array(resized_image)

  for i in range(128):
    for j in range(128):
      pixel = resized_image_array[i, j]
      if (pixel[0] <= threshold) or (pixel[1] <= threshold) or (pixel[2] <= threshold):
        resized_image_array[i,j] = [0,0,0]
      else:
        resized_image_array[i, j] = [255,255,255]

  resized_gray_image = Image.fromarray(resized_image_array)

  resized_gray_image.show()
  
  folder_path = "Inputs"
  # Generate a unique filename using uuid
  unique_filename = str(uuid.uuid4()) + ".png"
  path = os.path.join(folder_path, unique_filename)
  # Save the black and white image as a new file
  resized_gray_image.save(path)
  

def preprocess_image(input_image_path):
  
  """
  This function is used everytime we receive a new image as an input. It will transform it in a way that it looks like the ones in the dataset. 
  """
  
  original_image = Image.open(input_image_path)

  low_pass_image = original_image.filter(ImageFilter.GaussianBlur(radius=2))
  
  #Apply boundingbox
  # Calculate the bounding box coordinates
  non_zero_pixels = np.argwhere(np.array(low_pass_image)[:, :, 0] > 0)
  y1, x1 = np.min(non_zero_pixels, axis=0)
  y2, x2 = np.max(non_zero_pixels, axis=0)
  
  # Crop the image based on the bounding box
  cropped_image = low_pass_image.crop((x1, y1, x2, y2))
  
  # Resize the image to a 128x128
  resized_image = cropped_image.resize((128, 128), Image.LANCZOS)
  
  #Convert the image to a NumPy array, to access each pixel and change it to a greyscale
  image_array = np.array(resized_image)

  height, width, _ = image_array.shape

  #Make the image grayscale and use contrast
  threshold = 115
  for i in range(height):
    for j in range(width):
      pixel = image_array[i, j]
      red = pixel[0]
      green = pixel[1]
      blue = pixel[2]
      #Apply this formula to make it greyscale: 0.299 ∙ red + 0.587 ∙ green + 0.114 ∙ blue
      grayscale_value = int(0.2226 * red + 0.7152 * green + 0.0722 * blue)
      grayscale_value = max(0, min(255, grayscale_value))
      # make the pixel either perfect black or perfect white
      if grayscale_value <= threshold:
        image_array[i,j] = [0,0,0]
      else:
        image_array[i,j] = [255,255,255]
  #Convert the grayscale array back to an image format
  gray_image = Image.fromarray(image_array)
    
  folder_path = "Inputs"
  # Generate a unique filename using uuid
  unique_filename = str(uuid.uuid4()) + ".png"
  path = os.path.join(folder_path, unique_filename)
  # Save the black and white image as a new file
  gray_image.save(path)



#Balancing classes  

NUMBER_OF_CLASSES = 62

total_training_images = (34803 + 38049 + 34184 + 35293 + 33432 + 31067 + 34079 + 35796 + 
                33884 + 33720 + 7010 + 4091 + 11315 + 4945 + 5420 + 10203 + 2575 + 
                3271 + 13179 + 3962 + 2473 + 5390 + 10027 + 9149 + 28680 + 9277 + 
                2566 + 5436 + 23827 + 10927 + 14146 + 4951 + 5026 + 2731 + 5088 + 
                2698 + 11196 + 5551 + 2792 + 11421 + 28299 + 2493 + 3839 + 9713 + 
                2788 + 1920 + 2562 + 16937 + 2634 + 12856 + 2761 + 2401 + 3115 + 
                15934 + 2698 + 20793 + 2837 + 2854 + 2699 + 2820 + 2359 + 2726)

def get_average_images_trainingSet(total_images=total_training_images): 
  return (int) (total_images/ NUMBER_OF_CLASSES)

#print(get_average_images_trainingSet())

AVERAGE = 11801

def get_unbalanced_classes(path=BASE_PATH):
  unbalanced_classes = []
  for key, value in COUNT_DICT.items():
    if value <= 11801:
      unbalanced_classes.append(key)  
  return unbalanced_classes

#unbalanced_classes = get_unbalanced_classes()

#print(unbalanced_classes)


#DATA AUGMENTATION

#ROTATION

def augmentation_by_rotation(image_path, direction):
  img = cv2.imread(image_path)
  
  #Center of the image
  center = (128 // 2, 128 // 2)

  angle = 15 if direction == 'right' else -15
  
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  
  rotated_image = cv2.warpAffine(img, rotation_matrix, (128,128), borderValue=(255, 255, 255))
  
  cv2.imshow("Original Image", img)
  cv2.imshow("Rotated Image", rotated_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
#augmentation_by_rotation("Inputs\hsf_1_00016.png", 'left')

#Resizing

def augmentation_by_resizing(image_path):
  original_image = cv2.imread(image_path)
    
  resized_image = cv2.resize(original_image, (64, 64), interpolation=cv2.INTER_AREA)
  
  canvas = 255 * np.ones((128,128,3), dtype=np.uint8)
  
  x_offset = (128 - 64) // 2
  y_offset = (128 - 64) // 2
  
  canvas[y_offset:y_offset+64, x_offset:x_offset+64] = resized_image
  
    
  cv2.imshow("Origianl Image", original_image)
  cv2.imshow("Resized Image", canvas)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  

#augmentation_by_resizing("Inputs\hsf_1_00016.png")