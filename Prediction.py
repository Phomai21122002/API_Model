from PIL import Image
import numpy as np
import json
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

from keras.models import load_model

path_model = 'my_model.h5'
json_file_path = 'labels_mapping.json'

model = load_model(path_model)

with open(json_file_path, 'r') as json_file:
    labels_dict = json.load(json_file)

labels_dict = {v: k for k, v in labels_dict.items()}

def transform_image(image):
    image = image.resize((224, 224))  # Resize hình ảnh
    image_array = img_to_array(image) 
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array) 
    return preprocessed_image

def get_prediction(image_bytes):
  input_imgs = transform_image( image_bytes)
  predictions = model.predict([input_imgs])
  predicted_label_index  = np.argmax(predictions)
  accuracy_label = round(predictions[0][predicted_label_index] * 100, 2)
  predicted_label = labels_dict[predicted_label_index]
  return predicted_label, accuracy_label

def get_result(image_file):
  predicted_labels, accuracy_label = get_prediction(image_file)
  return predicted_labels, accuracy_label