from ultralytics import YOLO
import os

current_path = os.getcwd()

MODEL_PATH = "rpi_client/models/yolo11n-pose.pt"

MODEL_PATH = os.path.join(current_path, MODEL_PATH)

model = YOLO(MODEL_PATH)
# model.export(format="tflite", int8=True) 
# model.export(format="mnn", int8=True)
model.export(format="ncnn", half=True)