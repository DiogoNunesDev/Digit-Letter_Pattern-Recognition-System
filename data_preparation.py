import os
import numpy as np
from PIL import Image, ImageFilter
import uuid
import cv2

"""

THIS FILE IS RESPONSABLE FOR EVERYTHING CONCERNING DATA PREPROCESSING

"""

BASE_PATH = "C:\\Users\\diogo\\OneDrive\\Ambiente de Trabalho\Datasets\\by_class"


"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Auxiliar functions to prepare the dataset analysis

"""

def clean_mit_data(base_path=BASE_PATH):

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

def check_corruption(base_path=BASE_PATH):
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


"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Inputed data transformation section

"""

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

"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Balancing dataset classes section 

"""
NUMBER_OF_CLASSES = 62

def is_folder_below_threshold(file_count, threshold_min, threshold_max):
    return threshold_min <= file_count <= threshold_max 



#DATA AUGMENTATION -> MAKE SURE TO HAVE THE LIBRARIES USED INSTALED IN THE DATASET FOLDER!

#ROTATION

def augmentation_by_rotation(image_path, direction, img_name, folder_path):
  original_image = cv2.imread(image_path)
  #Center of the image
  center = (128 // 2, 128 // 2)

  angle = 15 if direction == 'right' else -15
  
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  
  rotated_image = cv2.warpAffine(original_image, rotation_matrix, (128,128), borderValue=(255, 255, 255))

  cv2.imwrite(folder_path +"\\"+ img_name + ".png", rotated_image)


#augmentation_by_rotation("Inputs\hsf_1_00016.png", 'left')
#augmentation_by_rotation("Inputs\hsf_1_00016.png", 'right')

#RESIZING

def augmentation_by_resizing(image_path, img_name, folder_path, factor):
  original_image = cv2.imread(image_path)
    
  resized_image = cv2.resize(original_image, (factor, factor), interpolation=cv2.INTER_AREA)
  
  canvas = 255 * np.ones((128,128,3), dtype=np.uint8)
  
  x_offset = (128 - factor) // 2
  y_offset = (128 - factor) // 2
  
  canvas[y_offset:y_offset+factor, x_offset:x_offset+factor] = resized_image
  _, bw_resized_image = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
  
  cv2.imwrite(folder_path +"\\"+ img_name + ".png", bw_resized_image)
  

#augmentation_by_resizing("Inputs\\train_4a_00029.png", 'resize_32', "Inputs", 32)
#augmentation_by_resizing("Inputs\hsf_1_00016.png", 32)

#TRANSLATION

def augmentation_by_translation(image_path, shift_x, shift_y, img_name, folder_path):
  original_image = cv2.imread(image_path)
  
  translation_shifting_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
  
  shifted_image = cv2.warpAffine(original_image, translation_shifting_matrix, (128, 128), borderValue=(255, 255, 255))
  
  cv2.imwrite(folder_path +"\\"+ img_name + ".png", shifted_image)

#augmentation_by_translation("Inputs\hsf_1_00016.png", 10, 10)
#augmentation_by_translation("Inputs\hsf_1_00016.png", -10, -10)
#augmentation_by_translation("Inputs\hsf_1_00016.png", -10, 10)
#augmentation_by_translation("Inputs\hsf_1_00016.png", 10, -10) 
 
#CHANGE PERSPECTIVE
def augmentation_by_perspective(image_path, perpesctive, factor, img_name, folder_path):
  original_image = cv2.imread(image_path)
  
  height, width = original_image.shape[:2]
  
  #Extract the 4 corners, this will be the reference points
  src_points = np.float32([[0,0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
  
  #Set the destination points
  if perpesctive == 'up': #Consists in moving the upward reference points a bit down so it creates a perspective
    dst_points = np.float32([[0, height * factor], [width - 1, height * factor], [0, height - 1], [width - 1, height - 1]]) 
    
  elif perpesctive == 'down': #Consists in moving the downward reference points a bit up so it creates a perspective
    dst_points = np.float32([[0, 0], [width - 1, 0], [0, height * (1 - factor)], [width - 1, height * (1 - factor)]])
  
  elif perpesctive == 'right':
    dst_points = np.float32([[0,0], [((width - 1) * (1 - factor)), 0], [0, height - 1], [((width - 1) * (1 - factor)), height - 1]])
  
  elif perpesctive == 'left':
    dst_points = np.float32([[width * factor ,0], [width-1, 0], [width * factor, height-1], [width-1, height-1]]) 
  
  #Get the perspective matrix based on both the source and destination points
  perpesctive_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
  
  perspective_image = cv2.warpPerspective(original_image, perpesctive_matrix, (128, 128), borderValue=(255,255,255))

  cv2.imwrite(folder_path +"\\"+ img_name + ".png", perspective_image)
  
#augmentation_by_perspective("Inputs\\train_41_00000.png", 'up', 0.5)  
#augmentation_by_perspective("Inputs\\train_41_00000.png", 'down', 0.5)  
#augmentation_by_perspective("Inputs\\train_41_00000.png", 'right', 0.5)  
#augmentation_by_perspective("Inputs\\train_41_00000.png", 'left', 0.5)  
  
#NOISE APPLICATION
def augmentation_by_noise(image_path, amount, img_name, folder_path):
  
  original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  
  height, width = original_image.shape[:2]
  
  noisy_image = np.copy(original_image)
   
  #Select random pixels from the image
  noise_pixels = []
  for i in range(amount):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    noisy_image[x, y] = 0
  
  cv2.imwrite(folder_path +"\\"+ img_name + ".png", noisy_image)
  
#augmentation_by_noise("Inputs\\train_4a_00029.png", 85)
  
#ELASTIFICATION
def augmentation_by_elastification(image_path, alpha, sigma, img_name, folder_path):
    """
    Parâmetros:
    - alpha: Distorcion intensity.
    - sigma: Distorcion smoothness.
    """
    original_image = cv2.imread(image_path)
    
    random_state = np.random.RandomState(None)

    shape = original_image.shape[:2]
    
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1

    # Suavizar os mapas de deslocamento com um desfoque gaussiano
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # Criar mapas de coordenadas
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # Aplicar a distorção elástica
    elastified_image = cv2.remap(original_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    
    _, elastified_bw_image = cv2.threshold(elastified_image, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(folder_path +"\\"+ img_name + ".png", elastified_bw_image)  
      
# Aplicar a distorção elástica
#augmentation_by_elastification("Inputs\\test_J.png", 50, 5, "elastico3", "Inputs")

"""
X = Number of files in folder -> possible transformations: 12
if x < 2500 -> *14 done
if 2500 > x < 3000 -> *10 done
if 3000 > x < 4000 -> *9 done
if 4000 > x < 5000 -> *7 done
if 5000 > x < 8000 -> *5 done
if 8000 > x < 9000 -> *4 done
if 9000 > x < 12000-> *3 done
if 12000 > x < 20000-> *2 done
"""

def implement_augmentation(dataset_path=BASE_PATH):
  for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    for sub_folder in os.listdir(folder_path):
      if sub_folder.startswith("train_"):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        augment(sub_folder, sub_folder_path)
        print(sub_folder + " is complete!")
  
  print("AUGMENTATION COMPLETE!")

def augment(folder, folder_path):
  count = len(os.listdir(folder_path))
  if is_folder_below_threshold(count, 0, 2500):
    
    for file in os.listdir(folder_path):
      #14 new images per image 
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'left', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 32)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, 10, 10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, -10, -10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, -10, 10, image_name, folder_path)
      count += 1

      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, 10, -10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_perspective(file_path, 'up', 0.5, image_name, folder_path)  
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_perspective(file_path, 'down', 0.5, image_name, folder_path)  
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_perspective(file_path, 'left', 0.5, image_name, folder_path)  
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_perspective(file_path, 'right', 0.5, image_name, folder_path)  
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_elastification(file_path, 50, 5, image_name, folder_path)
      count += 1
      
  elif is_folder_below_threshold(count, 2501, 3000):
    
    for file in os.listdir(folder_path):
      #10 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'left', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 32)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, 10, 10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, -10, -10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, -10, 10, image_name, folder_path)
      count += 1

      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_elastification(file_path, 50, 5, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_perspective(file_path, 'up', 0.5, image_name, folder_path)  
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1
      
  elif is_folder_below_threshold(count, 3001, 4000):
    
    for file in os.listdir(folder_path):
      #9 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'left', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 32)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, 10, 10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, -10, -10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_elastification(file_path, 50, 5, image_name, folder_path)
      count += 1

      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_perspective(file_path, 'up', 0.5, image_name, folder_path)  
      count += 1
      
  elif is_folder_below_threshold(count, 4001, 5000):
    
    for file in os.listdir(folder_path):
      #7 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'left', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, 10, 10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_elastification(file_path, 50, 5, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 32)
      count += 1

      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1
      
  elif is_folder_below_threshold(count, 5001, 8000):
    
    for file in os.listdir(folder_path):
      #5 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_elastification(file_path, 50, 5, image_name, folder_path)
      count += 1

      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_translation(file_path, 10, 10, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1

  elif is_folder_below_threshold(count, 8001, 9000):
    
    for file in os.listdir(folder_path):
      #4 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_elastification(file_path, 50, 5, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1

  elif is_folder_below_threshold(count, 9001, 12000):
    
    for file in os.listdir(folder_path):
      #3 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_resizing(file_path, image_name, folder_path, 64)
      count += 1

  elif is_folder_below_threshold(count, 12001, 20000):
    
    for file in os.listdir(folder_path):
      # Only 2 new images per image
      file_path = os.path.join(folder_path, file)
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_rotation(file_path, 'right', image_name, folder_path)
      count += 1
      
      image_name = folder + "_" + str(count).zfill(5)
      augmentation_by_noise(file_path, 85, image_name, folder_path)
      count += 1

"""
----------------------------------------------------------------------------------------------------------------------------------------------------
Funtion usage

"""

#clean_mit_data()

#check_corruption()

#implement_augmentation()


