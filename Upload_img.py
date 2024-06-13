import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import requests
import random
import shutil
from dotenv import dotenv_values

folder_img = "Data/"

def connect_cloudinary():
  env_vars = dotenv_values(".env")

  cloud_name = env_vars.get("CLOUD_NAME")
  api_key = env_vars.get("API_KEY")
  api_secret = env_vars.get("API_SECRET")

  cloudinary.config(
    cloud_name = cloud_name, 
    api_key = api_key, 
    api_secret = api_secret,
    secure = True
    )

def Upload_img_to_cloudinary(url, label):

  folder_upload = folder_img + label

  try:
    connect_cloudinary()

    response = cloudinary.uploader.upload(url, folder = folder_upload)
    return response["asset_id"], response["secure_url"]

  except Exception as e:
    print(e)

def call_api(url, root_folder):

  response = requests.get(url)
  data = response.json()

  for item in data:
    breed_name = item['cardBreedsId']['breed_name']
    image_url = item['links']  # Lấy hình ảnh đầu tiên

    if "Mèo" in breed_name or "mèo" in breed_name:
        breed_name = breed_name.replace("Mèo", "Cat").replace("mèo", "Cat")
    elif "Chó" in breed_name or "chó" in breed_name:
        breed_name = breed_name.replace("Chó", "Dogs").replace("chó", "Dogs")

    # Tạo tên thư mục dựa trên breed_name
    folder_name = os.path.join(root_folder, breed_name.replace(" ", "_"))
    
    # Tải và lưu hình ảnh
    download_image(image_url, folder_name)

def download_image(image_url, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
  
    image_name = os.path.join(folder_name, image_url.split('/')[-1])
    
    response = requests.get(image_url)
    with open(image_name, 'wb') as f:
        f.write(response.content)

def delete_folders_with_few_images(root_folder, min_images=50):
    for folder_name in os.listdir(root_folder):
      folder_path = os.path.join(root_folder, folder_name)
      
      if os.path.isdir(folder_path):
        num_images = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        if num_images < min_images:
          shutil.rmtree(folder_path)

def spl_images(input_folder, output_folder):
  # Tạo các thư mục train, test, val trong thư mục output
  train_folder = os.path.join(output_folder, 'Train')
  test_folder = os.path.join(output_folder, 'Test')
  val_folder = os.path.join(output_folder, 'Validation')

  os.makedirs(train_folder, exist_ok=True)
  os.makedirs(test_folder, exist_ok=True)
  os.makedirs(val_folder, exist_ok=True)

  # Lấy danh sách tất cả các thư mục (labels) trong thư mục input_folder
  labels = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

  for label in labels:
      label_folder = os.path.join(input_folder, label)
      image_files = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]

      # Trộn ngẫu nhiên danh sách ảnh
      random.shuffle(image_files)

      # Chia ảnh vào các tập train, test, val
      num_images = len(image_files)
      num_test = int(num_images * 0.15)
      num_val = int(num_images * 0.15)

      test_images = image_files[:num_test]
      val_images = image_files[num_test:num_val + num_test]
      train_images = image_files[num_val + num_test:]

      # Tạo thư mục tương ứng trong các thư mục train, test, val
      label_train_folder = os.path.join(train_folder, label)
      label_test_folder = os.path.join(test_folder, label)
      label_val_folder = os.path.join(val_folder, label)

      os.makedirs(label_train_folder, exist_ok=True)
      os.makedirs(label_test_folder, exist_ok=True)
      os.makedirs(label_val_folder, exist_ok=True)

      # Di chuyển các ảnh vào các thư mục tương ứng
      for image in train_images:
          shutil.copy(os.path.join(label_folder, image), os.path.join(label_train_folder, image))

      for image in test_images:
          shutil.copy(os.path.join(label_folder, image), os.path.join(label_test_folder, image))

      for image in val_images:
          shutil.copy(os.path.join(label_folder, image), os.path.join(label_val_folder, image))