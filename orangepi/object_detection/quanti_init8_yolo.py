from ultralytics import YOLO
model = YOLO('models/yolo11n-pose.pt')
# model.export(format="tflite", int8=True)
# model.export(format="mnn", int8=True) 
model.export(format="tflite", half=True, nms=True)