from collections import defaultdict
import os
import random
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""

THIS PYTHON FILE IS WHERE EVERYTHING RELATED TO DATA VISUALIZATION IS STORED

"""

BASE_PATH = "C:\\Users\\diogo\\OneDrive\\Ambiente de Trabalho\\Projects\\Sistema de Reconhecimento de Padroes\\byclass"

def create_sample_visualization(base_path, sample_per_folder=1):
    """
    This function returns a dictionary where the keys are the folder names and the values are a list of the sample image paths based on that folder.
    Only training data will be visualized so this function only retrieves data from training folders.
    """
    samples = defaultdict(list)

    # Iterate over the folders in the base path
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        # iterate over the Character folders
        for sub_folder in os.listdir(folder_path):
            # If the folder is a training folder (this will be our target folder)
            if sub_folder.startswith("train_"):
                train_folder_path = os.path.join(folder_path, sub_folder)
                # iterate over the files in the target folder and add their paths to a list
                image_list=[]
                for file in os.listdir(train_folder_path):
                    #if it is a png image
                    if file.lower().endswith('.png'):
                        image_list.append(os.path.join(train_folder_path, file))
            
                samples[train_folder_path] = random.sample(image_list, min(sample_per_folder, len(image_list)))
    return samples


def visualize_sample_training_data(data):
    rows = 11  # Número total de linhas
    columns = 6  # Número de colunas (imagens por linha)

    fig, axs = plt.subplots(rows, columns, figsize=(25, 55))

    image_counter = 0
    for folder_name, image_paths in data.items():
        for image_path in image_paths:
            path = folder_name.split('\\byclass\\')
            row = image_counter // columns
            col = image_counter % columns
            img = mpimg.imread(image_path)
            axs[row, col].imshow(img)
            title = chr(int(path[1][0:2], 16))
            axs[row, col].set_title("Label: " + title + "", y=-0.15)
            axs[row, col].axis('off')
            #Make borders visible
            for spine in axs[row, col].spines.values():
                spine.set_visible(True)

            image_counter += 1
            #make the axis with no images off
            for i in range(2, 6): 
                axs[-1, i].axis('off')

            if image_counter == 62:
                break

    plt.tight_layout()
    plt.show()

def count_data_per_training_folder(path=BASE_PATH):

    count_dict= {}
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for sub_folder in os.listdir(folder_path):
            if sub_folder.startswith("train_"):
                train_folder_path = os.path.join(folder_path, sub_folder)
                items = os.listdir(train_folder_path)
                count_dict[sub_folder] = len(items)
    
    return count_dict


def print_dict(dict):
    for key, value in dict.items():
        print(f'{key}: {value}')

#samples = create_sample_visualization(BASE_PATH, 1)

#visualize_sample_training_data(samples)

#COUNT_DICT = count_data_per_training_folder()

