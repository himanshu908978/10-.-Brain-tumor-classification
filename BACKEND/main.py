from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from model import inference
import os

labels = ['notumor', 'tumor']
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.post('/braintumor')
async def classifier(data:UploadFile = File(...)):
    file_location = f"temp_{data.filename}"

    with open(file_location,"wb") as buffer:
        buffer.write(await data.read())

    pred_class,conf = inference(file_location)
    conf = round(conf,4)
    os.remove(file_location)

    return {
        "pred_label":labels[pred_class],
        "conf":conf*100
    }