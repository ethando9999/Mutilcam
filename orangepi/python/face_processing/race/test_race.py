# test_gender.py
# from landmark_face import LandmarkFace
from race_predict import RacePredictor  # Giả định GenderPredictor được lưu trong file gender_predictor.py
import cv2
import time
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

image_path = "a_phong.jpg" # Có thể thay bằng ảnh khác 


def main():
    # face_detection = LandmarkFace()
    race_predictor = RacePredictor()
    image = cv2.imread(image_path)
    start_time = time.time()

    race, confidence = race_predictor.predict_race(image)
    # Tính toán FPS mỗi giây
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    logging.info(f"Race: {race}")
    logging.info(f"confidence: {confidence}") 
    logging.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()