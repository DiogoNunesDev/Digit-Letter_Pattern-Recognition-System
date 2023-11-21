import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import uuid
import cv2

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




#Inputed data transformation funtion

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
  



