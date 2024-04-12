from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from starlette.responses import RedirectResponse
from PIL import Image
import io
import Prediction

app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/api/prediction")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    
    try:
      bytes_io = io.BytesIO(await file.read())
      
      image = Image.open(bytes_io)
      result_label, result_accuracy = Prediction.get_result(image)
      
      return {"label": result_label, "accuracy" : result_accuracy}
  
    except Exception as e:
        # Trả về lỗi nếu có bất kỳ lỗi nào xảy ra
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
  uvicorn.run(app, port=8000, host='127.0.0.1')