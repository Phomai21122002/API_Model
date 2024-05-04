import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import dotenv_values

folder_img = "Data/"

def Upload_img_to_cloudinary(url, label):

  folder_upload = folder_img + label
  print(folder_upload)

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

    response = cloudinary.uploader.upload(url, folder = folder_upload)
    return response["asset_id"], response["secure_url"]

  except Exception as e:
    print(e)