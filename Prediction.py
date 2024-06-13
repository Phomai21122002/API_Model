# from PIL import Image
import numpy as np
import json
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
import h5py
from pymongo import MongoClient
from PIL import Image
from fastapi import HTTPException
import re
from bson import ObjectId
from datetime import datetime
from dotenv import dotenv_values


env_vars = dotenv_values(".env")
path_db = env_vars.get("PATH_DB")

def load_model_h5():
  path_model = './file_export/my_model_NetV2.h5'
  json_file_path = './file_export/labels.json'

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

  return model, labels_dict

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

def get_prediction(image_bytes, model, labels_dict):
  input_imgs = transform_image( image_bytes)
  predictions = model.predict([input_imgs])
  predicted_label_index  = np.argmax(predictions)
  accuracy_label = round(predictions[0][predicted_label_index] * 100, 2)
  predicted_label = labels_dict[predicted_label_index]
  return predicted_label, accuracy_label, predicted_label_index

def get_result(image_file, model, labels_dict):
  predicted_labels, accuracy_label, predicted_label_index = get_prediction(image_file, model, labels_dict)
  return predicted_labels, accuracy_label, predicted_label_index

def convert_to_json(obj):
    if isinstance(obj, ObjectId):  # Nếu đối tượng là ObjectId, chuyển thành chuỗi
        return str(obj)
    elif isinstance(obj, datetime):  # Nếu đối tượng là datetime, chuyển thành chuỗi đại diện thời gian
        return obj.isoformat()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def data_db(result_label):
  try:
    db = connect_db(path_db)
    collection = db["cardBreed"]

    name = result_label.split("_")[1:]
    name = ' '.join(name)
    name = re.compile(name, re.IGNORECASE)

    pipeline = [
        {
            "$match": {"breed_name": {"$regex": name}}  # Chỉ lấy thông tin của thú cưng có _id = 1
        },
        {
            "$lookup": {
                "from": "product",
                "localField": "diet",
                "foreignField": "_id",
                "as": "diets"
            }
        },
        {
          "$project": {
              "__v": 0,
              "htmlDomDescription": 0,
              "diets.__v": 0,
              "diets.htmlDomDescription": 0,
              "diet": 0
          }
      }
    ]

    data_breed = list(collection.aggregate(pipeline))
    if not data_breed:
      return {}
    data_breed = json.dumps(data_breed, indent=4, default=convert_to_json)
    return json.loads(data_breed)[0]
  except Exception as e:
      # Xử lý lỗi và in ra thông báo lỗi
      error_message = {"error": str(e)}
      return json.dumps(error_message)

def process_image(bytes_io):
  try:
    image = Image.open(bytes_io)

    model, labels_dict = load_model_h5()
    result_label, result_accuracy, result_id = get_result(image, model, labels_dict)
    data_breed = data_db(result_label)
    return result_id, result_label, result_accuracy, data_breed

  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
    