from ultralytics import YOLO
model = YOLO("yolo11s-pose.pt")  # Thay bằng tên mô hình YOLO Pose của bạn
model.export(format="rknn")