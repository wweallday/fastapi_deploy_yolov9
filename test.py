import cv2 
import numpy
import os
from enum import Enum
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# dir_name = "images_upload"
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)
# app = FastAPI(title= 'deploy a yolo model with fastapi!')

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes)
    # print("hello",result)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


img = cv2.imread('cars.png')
model = YOLO('yolov9c.pt')
result_img, _ = predict_and_detect(chosen_model=model, img=np.copy(img),classes=[2],conf=0.5)

cv2.imwrite(f'images_upload/predict.png', result_img)


# @app.get("/")
# async def home():
#     return {"message":"your fastapi is working!"}

# @app.post("/predict")
# async def predict(model: Model,file: UploadFile = File(...)):
#     filename = file.filename
#     file_type = filename.split(".")[-1] in ("jpg","jpeg","png")
#     if not file_type:
#         raise HTTPException(status_code=415, detail="Unsupported file provided.")

#     #read image
#     image_stream = io.BytesIO(file.file.read())
#     #start the stream from the position zero
#     image_stream.seek(0)
#     #write the stream of bytes into array
#     file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
#     #decode image
#     image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
#     #run the model to detect car in coco 
#     result_img, _ = predict_and_detect(model, image.copy(),classes=[2])
    
#     base_name, extension = os.path.splitext(filename)
#     new_filename = f"{base_name}_predicted{extension}"

#     # Save the image with the new filename
#     cv2.imwrite(f'images_uploaded/{new_filename}', result_img)

#     file_image = open(f'images_uploaded/{new_filename}', mode="rb")
    
#     # Return the image as a stream specifying media type
#     return StreamingResponse(new_filename, media_type="image/jpeg")