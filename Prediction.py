# from PIL import Image
import numpy as np
import json
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from keras.models import load_model
import h5py

print(tf.__version__)
path_model = 'my_model_NetV2.h5'
json_file_path = 'labels.json'

# Load the model
with h5py.File(path_model, 'r+') as f:
  if 'model_config' in f.attrs:
    config = f.attrs['model_config']
    model_config = json.loads(config)
    for layer in model_config['config']['layers']:
      if layer['class_name'] == 'DepthwiseConv2D':
        # Remove the 'groups' parameter if it exists
        if 'groups' in layer['config']:
          del layer['config']['groups']
    # Update the model configuration in the file
    f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

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
  print(predicted_label_index)
  accuracy_label = round(predictions[0][predicted_label_index] * 100, 2)
  predicted_label = labels_dict[predicted_label_index]
  return predicted_label, accuracy_label, predicted_label_index

def get_result(image_file):
  predicted_labels, accuracy_label, predicted_label_index = get_prediction(image_file)
  return predicted_labels, accuracy_label, predicted_label_index
