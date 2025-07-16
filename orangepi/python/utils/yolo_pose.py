from ultralytics import YOLO
import cv2
import numpy as np
import os
from .logging_python_orangepi import get_logger
import time

logger = get_logger(__name__)

current_path = os.getcwd()


# MODEL_PATH = "models/yolo11n-pose_ncnn_model"
MODEL_PATH = "models/yolo11n-pose_rknn_model"

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODEL_PATH)


COCO_EDGES = [
    (0, 1),  # Mũi đến mắt trái
    (0, 2),  # Mũi đến mắt phải
    (1, 3),  # Mắt trái đến tai trái
    (2, 4),  # Mắt phải đến tai phải
    (5, 6),  # Vai trái đến vai phải
    (5, 7),  # Vai trái đến khuỷu tay trái
    (7, 9),  # Khuỷu tay trái đến cổ tay trái
    (6, 8),  # Vai phải đến khuỷu tay phải
    (8, 10), # Khuỷu tay phải đến cổ tay phải
    (11, 12),# Hông trái đến hông phải
    (11, 13),# Hông trái đến đầu gối trái
    (13, 15),# Đầu gối trái đến mắt cá chân trái
    (12, 14),# Hông phải đến đầu gối phải 
    (14, 16),# Đầu gối phải đến mắt cá chân phải
    (5, 11), # Vai trái đến hông trái
    (6, 12), # Vai phải đến hông phải
    (6, 11)  # Vai phải đến hông trái (đường chéo cơ thể)
]

class HumanDetection:
    # def __init__(self, device="cpu"): 
    #     self.device = device
    def __init__(self):
        logger.info('Init Human Detecton')
        self.classes = [0]  # 0 for human
        # Load the YOLO model once during initialization
        self.model = YOLO(MODEL_PATH)
        self.fps_avg = 0.0
        self.call_count = 0

    def run_detection(self, source):
        """
        Run human detection on the given video source. 

        Args:
            source (str): Path to the video or image file.
        
        Returns:
            keypoints_data (np.ndarray): Keypoints detected for each human.
            boxes_data (list of tuples): Bounding boxes for detected humans. 
        """
        start_time = time.time()
        results = self.model.predict(
            source=source,
            verbose=False,
            classes=self.classes,
            # device=self.device,
            conf=0.5,
            iou=0.4 
        )
        self.results = results[0] 

        # Tính FPS
        end_time = time.time()
        duration = end_time - start_time
        fps_current = 1 / duration if duration > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1
        logger.info(f"FPS Human detection: {self.fps_avg:.2f}")
        
        if not results:  # Handle empty results
            return np.array([]), []

        # Extract keypoints and bounding boxes 
        keypoints_data = results[0].keypoints.xy.cpu().numpy()
        boxes_data = [
            tuple(map(int, box.xyxy[0].tolist())) for box in results[0].boxes
        ]

        return keypoints_data, boxes_data
    
    def draw_boxes_and_edges(self):
        # Plot results with filtered boxes
        annotated_img = self.results.plot()
        return annotated_img
    
    def transform_keypoints_to_local(self, box, keypoints):
        """
        Chuyển đổi keypoints từ ảnh gốc sang tọa độ trong bounding box.
        
        box: Tuple (x1, y1, x2, y2) - Bounding box
        keypoints: List [(x, y), (x, y), ...] - Danh sách keypoints của box này
        
        Returns: List of transformed keypoints [(x', y'), (x', y'), ...]
        """
        x1, y1, _, _ = box
        transformed_keypoints = []

        for (x, y) in keypoints:
            # Chuyển đổi tọa độ
            new_x = x - x1
            new_y = y - y1
            
            # Nếu x hoặc y ban đầu bằng 0, thì x' hoặc y' cũng bằng 0
            if x == 0:
                new_x = 0
            if y == 0:
                new_y = 0

            transformed_keypoints.append((new_x, new_y))

        return transformed_keypoints




if __name__ == "__main__":
    detector = HumanDetection()  # Use GPU if available
    source = cv2.VideoCapture("output_4k_video.mp4") 
    while True:
        ret, frame = source.read()
        if not ret:
            print("Không thể đọc frame từ video")
            break
        
        # Phát hiện người và keypoints 
        keypoints, boxes = detector.run_detection(frame)