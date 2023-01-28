from fastapi import FastAPI, File, UploadFile
from PIL import Image
from img2tex.predict import predict
from fastapi.staticfiles import StaticFiles

app = FastAPI()


@app.post("/uploadfile/{scale}")
async def create_upload_file(scale:float ,files: UploadFile):
    result = predict(files.file, scale)
    return {"filename": files.filename, "content":files.content_type,"math":result}

# @app.get("/")
# async def read_main():
#     return {"msg": "Hello World"}