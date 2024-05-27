import cv2 
import numpy
import os
from enum import Enum
import io
import uvicorn
import numpy as np
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

dir_name = "images_upload"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

app = FastAPI(title= 'deploy a yolo model with fastapi!')

def predict_out(chosen_model, img, classes=[], conf=0.5):
    # print(chosen_model)
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict_out(chosen_model, img, classes,conf=0.5)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

class Model(str, Enum):
    yolo = 'yolov9c.pt'

@app.get("/")
async def home():
    return {"message":"your fastapi is working!"}

# @app.post("/predict")
# async def predict(model: Model,file: UploadFile = File(...)):
#     filename = file.filename
#     file_type = filename.split(".")[-1] in ("jpg","jpeg","png")
#     if not file_type:
#         raise HTTPException(status_code=415, detail="Unsupported file provided.")
#     yolo_model = YOLO(model.yolo.value)
#     #read image
#     image_stream = io.BytesIO(file.file.read())
#     #start the stream from the position zero
#     image_stream.seek(0)
#     #write the stream of bytes into array
#     file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
#     #decode image
#     image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
#     #run the model to detect car in coco 
#     print("hellllll",len(image))
#     result_img, _ = predict_and_detect(yolo_model, image.copy(),classes=[2],conf=0.5)
    
#     base_name, extension = os.path.splitext(filename)
#     new_filename = f"{base_name}_predicted{extension}"
#     print(new_filename)
#     # Save the image with the new filename
#     cv2.imwrite(f'images_upload/{new_filename}', result_img)

#     file_image = open(f'images_upload/{new_filename}', mode="rb")
    
#     # Return the image as a stream specifying media type
#     return StreamingResponse(file_image, media_type="image/jpeg")


@app.post("/predict")
async def predict(model: Model,file: UploadFile = File(...)):
    filename = file.filename
    file_type = filename.split(".")[-1] in ("jpg","jpeg","png")
    if not file_type:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    #read image
    image_stream = io.BytesIO(file.file.read())
    #start the stream from the position zero
    image_stream.seek(0)
    #write the stream of bytes into array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    #decode image
    image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
    #run the model to detect car in coco 
    yolo_model = YOLO(model.yolo.value)

    result_img, _ = predict_and_detect(yolo_model, image.copy(),classes=[2], conf=0.5)
    cv2.imwrite(f'images_upload/{filename}', result_img)

    file_image = open(f'images_upload/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")