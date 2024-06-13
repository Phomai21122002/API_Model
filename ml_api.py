from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn
from starlette.responses import RedirectResponse
import io
import threading
import requests
import time
from fastapi.middleware.cors import CORSMiddleware
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import Prediction
import update_model
import Upload_img

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Accept-Encoding", "Authorization", "Content-Type"],
)

model = Prediction.load_model_h5()  # Function to load initial model

class ModelReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('./file_export/my_model_NetV2.h5'):
            global model
            model = Prediction.load_model_h5()  # Function to reload model
            print("Model reloaded")

def start_observer():
    event_handler = ModelReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/api/prediction")
async def predict_api(file: UploadFile = File(None), file_url: str = Form(None)):
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

        result_id, result_label, result_accuracy, data_breed = Prediction.process_image(bytes_io)

        # upload img to cloudinary
        upload_result = Upload_img.Upload_img_to_cloudinary(contents, result_label)
        if upload_result is not None:
          id_img, url_img = upload_result
        else:
          return {"message": "Upload image return None"}
        return {"id": int(result_id), "label": result_label, "accuracy": result_accuracy, "id_img": id_img, "url": url_img, "data_breed": data_breed}
    
    if file_url:
      # Tải hình ảnh từ URL trực tiếp
      try:
          response = requests.get(file_url)
          response.raise_for_status()
          bytes_io = io.BytesIO(response.content)
          # predict img
          result_id, result_label, result_accuracy, data_breed = Prediction.process_image(bytes_io)
          # upload img to cloudinary
          upload_result = Upload_img.Upload_img_to_cloudinary(file_url, result_label)
          if upload_result is not None:
            id_img, url_img = upload_result
          else:
            return {"message": "Upload image return None"}
          
          return {"id": int(result_id), "label": result_label, "accuracy": result_accuracy, "id_img": id_img, "url": url_img, "data_breed": data_breed}
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/update_new_model")
async def update_new_model():
    try:
      # download img from api feedback
      url = "http://localhost:3000/api/feedbacks"
      root_folder = "Data"
      Upload_img.call_api(url, root_folder)
      Upload_img.delete_folders_with_few_images(root_folder)

      # Split data img to train, test, validation
      path = './Data/'
      path_spl = './Dataspl/'
      Upload_img.spl_images(path, path_spl)

      # run update model
      threading.Thread(target=update_model.file_update_model).start()
    except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
    return "successful"

# @app.get("/download_img")
# async def download_img():
#     try:
#       url = "http://localhost:3000/api/feedbacks"
#       root_folder = "Phomai"
#       Upload_img.call_api(url, root_folder)

#       Upload_img.delete_folders_with_few_images(root_folder)

#       # # Split Data img to train, test, validation
#       path = './Phomai/'
#       path_spl = './PhomaiSpl/'
#       Upload_img.spl_images(path, path_spl)

#       threading.Thread(target=update_model.file_update_model).start()

#     except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
#     return "successful"

if __name__ == '__main__':
  observer_thread = threading.Thread(target=start_observer)
  observer_thread.daemon = True
  observer_thread.start()
  uvicorn.run(app, port=8000, host='127.0.0.1')