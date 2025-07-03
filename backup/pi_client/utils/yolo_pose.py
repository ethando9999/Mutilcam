# from ultralytics import YOLO
# import cv2
# import numpy as np

# MODEL_PATH = "pi_client/models/yolo11n-pose_int8_openvino_model"

# COCO_EDGES = [
#     (0, 1), (0, 2), (1, 3), (2, 4),  # Head and facial connections
#     (5, 6),  # Shoulders
#     (5, 7), (7, 9),  # Left arm
#     (6, 8), (8, 10),  # Right arm
#     (11, 12),  # Hips
#     (11, 13), (13, 15),  # Left leg
#     (12, 14), (14, 16),  # Right leg
#     (5, 11), (6, 12),  # Torso
#     (6, 11)  # Body
# ]

# class HumanDetection:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.classes = [0]  # 0 for human
#         # Load the YOLO model once during initialization
#         self.model = YOLO(MODEL_PATH)

#     def run_detection(self, source):
#         """
#         Run human detection on the given video source.

#         Args:
#             source (str): Path to the video or image file.
        
#         Returns:
#             keypoints_data (np.ndarray): Keypoints detected for each human.
#             boxes_data (list of tuples): Bounding boxes for detected humans.
#         """
#         results = self.model.predict(
#             source=source,
#             verbose=False,
#             classes=self.classes,
#             device=self.device,
#             conf=0.3,
#             iou=0.5
#         )
#         self.results = results[0]
        
#         if not results:  # Handle empty results
#             return np.array([]), []

#         # Extract keypoints and bounding boxes
#         keypoints_data = results[0].keypoints.xy.cpu().numpy()
#         boxes_data = [
#             tuple(map(int, box.xyxy[0].tolist())) for box in results[0].boxes
#         ]

#         return keypoints_data, boxes_data
    
#     def draw_boxes_and_edges(self):
#         # Plot results with filtered boxes
#         annotated_img = self.results.plot()
#         return annotated_img

# # Example usage:
# # detector = HumanDetection(device="cuda")  # Use GPU if available
# # keypoints, boxes = detector.run_detection("test_video/test.mp4")
