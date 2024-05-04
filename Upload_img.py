import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import dotenv_values
import cv2
from io import BytesIO

folder_img = "Data/"

def Upload_img_to_cloudinary(image_data, label):

  folder_upload = folder_img + label

  try:
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

    # Convert image data to JPEG format
    ret, buf = cv2.imencode('.jpg', image_data)
    encoded_image = BytesIO(buf)

    # Upload the image to Cloudinary
    response = cloudinary.uploader.upload(encoded_image, folder=folder_upload)
    return response["asset_id"], response["secure_url"]

  except Exception as e:
    print(e)