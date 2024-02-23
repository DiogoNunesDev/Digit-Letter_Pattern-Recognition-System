import os
import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
from src.data_preparation import preprocess_image
import tensorflow as tf
import numpy as np

"""

THIS PYTHON FILE IS THE INTERFACE OF THE SYSTEM

"""

# GUI basic setup
window = tk.Tk()
window.title("Pattern Recognition System")
window.geometry("500x500")

image_frame = tk.Frame(window)
image_frame.pack(side=tk.TOP, expand=True)

file_path = None

#Show selected image on screen
def submit_image():
  global file_path
  file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
  if file_path:
      # Load the image using PIL
      img = Image.open(file_path)
      img.thumbnail((400, 400))  # Resize the image to fit in the GUI
      img = ImageTk.PhotoImage(img)

      # If an image is already displayed, remove it
      for widget in image_frame.winfo_children():
          widget.destroy()

      # Display the image
      image_label = tk.Label(image_frame, image=img)
      image_label.image = img  # Keep a reference to the image
      image_label.pack()


def get_class(predictions):
  index = np.argmax(predictions, axis=1)[0]
  print(index)
  char_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  
  return char_labels[index]

#Selected image preprocessing and calculations
def process_image():
  img_array = preprocess_image(file_path)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255
  print(img_array.shape)
  model = tf.keras.models.load_model('CNN_model_V1.h5')
  predictions = model.predict(img_array)
  prediction = get_class(predictions)
  prediction_label.config(text=f"Predicted Class: {prediction}")

#GUI Setup
button_font = font.Font(family="Courier New", size=12, weight="bold")
button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM)

submit_btn = tk.Button(button_frame, text="Submit Image", command=submit_image, width=20, height=2, font=button_font, borderwidth=4, relief=tk.RAISED)
submit_btn.pack(side=tk.LEFT, pady=5, padx=20)

process_btn = tk.Button(button_frame, text="Process Image", command=process_image, width=20, height=2, font= button_font, borderwidth=4, relief=tk.RAISED)
process_btn.pack(side=tk.RIGHT, pady=5, padx=20)

prediction_label = tk.Label(window, text=" ", font=("Courier New", 14))
prediction_label.pack(side=tk.BOTTOM, pady=0)

window.mainloop()