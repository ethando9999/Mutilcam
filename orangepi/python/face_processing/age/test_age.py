# from landmark_face import LandmarkFace
from age_predict import AgePredictor
import cv2
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# image_path = "calibration_data/397.jpg"
image_path = "a_phong.jpg"
 
def main():
    # face_detection = LandmarkFace()
    age_predictor = AgePredictor()
    image = cv2.imread(image_path)

    frame_count = 0
    start_time = time.time()

    while True:
        age_range, confidence = age_predictor.predict_age(image)
        frame_count += 1

        # Tính toán FPS mỗi giây
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            logging.info(f"Age: {age_range}")
            logging.info(f"confidence: {confidence}") 
            logging.info(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

def main2():
    # face_detection = LandmarkFace()
    age_predictor = AgePredictor()
    image = cv2.imread(image_path)
    start_time = time.time()

    age_range, confidence = age_predictor.predict_age(image)
    # Tính toán FPS mỗi giây
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    logging.info(f"Age: {age_range}")
    logging.info(f"confidence: {confidence}") 
    logging.info(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main2()