import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('C:\\Users\\Shree\\Desktop\\projects\\BTD\\archive (1)\\pred\\pred5.jpg')

if image is not None:  # Check if image was read successfully
  img = Image.fromarray(image)
  img = img.resize((64,64))

  img = np.array(img)

  input_img = np.expand_dims(img, axis=0)

  result = model.predict(input_img)
  predicted_class = np.argmax(result, axis=1)  # Get the index of the highest probability class
  print(predicted_class)
else:
  print("Error: Image could not be read. Check the image path or format.")
