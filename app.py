import os
import io
import cv2
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# Directory to save uploaded images
dir_name = "images_upload"

app = FastAPI(title='Deploy a YOLO Model with FastAPI!')

# Ensure the upload directory exists
@app.on_event("startup")
async def startup_event():
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# Enum for model selection
class Model(str, Enum):
    yolo = 'yolov9c.pt'
    yolov8n = 'yolov8n.pt'

# Function to predict and return results
def predict_out(chosen_model, img, classes=[], conf=0.5):
    results = chosen_model.predict(img, classes=classes, conf=conf) if classes else chosen_model.predict(img, conf=conf)
    return results

# Function to draw predictions on the image
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict_out(chosen_model, img, classes, conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

@app.get("/")
async def home():
    return {"message": "Your FastAPI is working!"}

@app.post("/predict")
async def predict(model: Model, file: UploadFile = File(...)):
    filename = file.filename
    file_extension = filename.split(".")[-1]
    if file_extension not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    try:
        # Read and decode the image
        image_stream = io.BytesIO(file.file.read())
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # Load the YOLO model
    print(model)
    yolo_model = YOLO(model.value)

    # Run the model and detect objects
    result_img, _ = predict_and_detect(yolo_model, image.copy(), classes=[2], conf=0.5)
    output_path = os.path.join(dir_name, filename)
    cv2.imwrite(output_path, result_img)

    # Return the processed image as a stream
    return StreamingResponse(io.BytesIO(open(output_path, "rb").read()), media_type="image/jpeg")