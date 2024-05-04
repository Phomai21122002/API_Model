from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn
from starlette.responses import RedirectResponse
from PIL import Image
import io
import Prediction
import Upload_img
import DetectObject
import requests
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):  # Định nghĩa một lớp BaseModel cho request body
    file_url: str | None = ''

async def process_image(image):
    try:
      result_label, result_accuracy, result_id = Prediction.get_result(image)

      return result_id, result_label, result_accuracy

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/api/prediction")
async def predict_api(file: UploadFile = File(None), file_url: str = Form(None)):
    objectSplImage = []

    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Please provide either a file or a file URL")
    
    if file and file_url:
        raise HTTPException(status_code=400, detail="Please provide either a file or a file URL, not both")

    if file:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
          raise HTTPException(status_code=400, detail="Image must be jpg or png format!")
        
        contents = await file.read()
        # Nếu đầu vào là một file được tải lên, xử lý hình ảnh từ file
        bytes_io = io.BytesIO(contents)

        countImgSpl = DetectObject.DetectAnimal(bytes_io)

        if(len(countImgSpl) > 0):
          for img in countImgSpl:
             # predict img
            result_id, result_label, result_accuracy = await process_image(img)

            # upload img to cloudinary
            upload_result = Upload_img.Upload_img_to_cloudinary(img, result_label)
            if upload_result is not None:
              id_img, url_img = upload_result
            else:
              return {"message": "Upload image return None"}
            
            objectSplImage.append({"id": int(result_id), "label": result_label, "accuracy": result_accuracy, "id_img": id_img, "url": url_img})
        
        objectSplImage = sorted(objectSplImage, key=lambda x: x['accuracy'], reverse=True)

        return objectSplImage
    
    if file_url:
      if not ((file_url.startswith("http://") or file_url.startswith("https://")) and file_url.split(".")[-1] in ("jpg", "jpeg", "png")):
        raise HTTPException(status_code=400, detail="Invalid file URL")
        
      # Tải hình ảnh từ URL trực tiếp
      try:
          response = requests.get(file_url)
          response.raise_for_status()
          bytes_io = io.BytesIO(response.content)
          
          countImgSpl = DetectObject.DetectAnimal(bytes_io)

          if(len(countImgSpl) > 0):
            for img in countImgSpl:
              # predict img
              result_id, result_label, result_accuracy = await process_image(img)

              # upload img to cloudinary
              upload_result = Upload_img.Upload_img_to_cloudinary(img, result_label)
              if upload_result is not None:
                id_img, url_img = upload_result
              else:
                return {"message": "Upload image return None"}
              
              objectSplImage.append({"id": int(result_id), "label": result_label, "accuracy": result_accuracy, "id_img": id_img, "url": url_img})
          
          objectSplImage = sorted(objectSplImage, key=lambda x: x['accuracy'], reverse=True)

          return objectSplImage
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
  uvicorn.run(app, port=8000, host='127.0.0.1')