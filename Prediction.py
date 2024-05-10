# from PIL import Image
import numpy as np
import json
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from keras.models import load_model
import h5py
from pymongo import MongoClient
from PIL import Image
from fastapi import HTTPException
import re
from bson import ObjectId
from datetime import datetime

print(tf.__version__)
path_model = 'my_model_NetV2.h5'
json_file_path = 'labels.json'

path_db = "mongodb+srv://nhatlinhdut3:td1uAMAgupGhminV@pbl7.ozxnm0y.mongodb.net/test?retryWrites=true&w=majority&appName=pbl7"

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

# db mongoDB
def connect_db(path_connect):
  client = MongoClient(path_connect)
  db = client["test"]
  return db

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

def convert_to_json(obj):
    if isinstance(obj, ObjectId):  # Nếu đối tượng là ObjectId, chuyển thành chuỗi
        return str(obj)
    elif isinstance(obj, datetime):  # Nếu đối tượng là datetime, chuyển thành chuỗi đại diện thời gian
        return obj.isoformat()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def data_db(result_label):
  db = connect_db(path_db)
  collection = db["cardBreed"]

  name = result_label.split("_")[1:]
  name = ' '.join(name)
  name = re.compile(name, re.IGNORECASE)

  projection = {"__v": 0, "htmlDomDescription": 0}
  query = {"breed_name": {"$regex": name}}
  data_breed = collection.find_one(query, projection)
  data_breed = json.dumps(data_breed, default=convert_to_json)
  
  return data_breed

def process_image(bytes_io):
  try:
    image = Image.open(bytes_io)
    result_label, result_accuracy, result_id = get_result(image)
    data_breed = data_db(result_label)
    data_breed = json.loads(data_breed)
    return result_id, result_label, result_accuracy, data_breed

  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
    