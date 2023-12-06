import os
import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
from data_preparation import preprocess_image
import tensorflow as tf


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

#Selected image preprocessing and calculations
def process_image():
  img = preprocess_image(file_path)
  model = tf.keras.models.load_model('CNN_model.h5')
  model.predict(img)


#GUI Setup
button_font = font.Font(family="Courier New", size=12, weight="bold")
button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM)

submit_btn = tk.Button(button_frame, text="Submit Image", command=submit_image, width=20, height=2, font=button_font, borderwidth=4, relief=tk.RAISED)
submit_btn.pack(side=tk.LEFT, pady=20, padx=20)

process_btn = tk.Button(button_frame, text="Process Image", command=process_image, width=20, height=2, font= button_font, borderwidth=4, relief=tk.RAISED)
process_btn.pack(side=tk.RIGHT, pady=20, padx=20)

window.mainloop()