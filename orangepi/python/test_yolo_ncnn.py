
import logging
from utils.yolo_pose import HumanDetection
import cv2 


detector = HumanDetection()  # Use GPU if available
source = cv2.VideoCapture("python/data/output_4k_video.mp4") 
while True:
    ret, frame = source.read()
    if not ret:
        print("Không thể đọc frame từ video")
        break
    
    # Phát hiện người và keypoints 
    keypoints, boxes = detector.run_detection(frame) 
    print("ok!") 