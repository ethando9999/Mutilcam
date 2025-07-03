from ultralytics import YOLO
model = YOLO('rust/yolo_tflite/models/yolo11n-pose.pt')
model.export(format="tflite", half=True) 