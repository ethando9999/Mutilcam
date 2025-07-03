# test_gender.py
# from landmark_face import LandmarkFace
from gender_predictor import GenderPredictor  # Giả định GenderPredictor được lưu trong file gender_predictor.py
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

# Đường dẫn đến file mô hình RKNN và ảnh test
MODEL_PATH = "python/face_proccesing/models/man_woman_face_vit.rknn"
image_path = "4686x4000.jpg"  # Có thể thay bằng ảnh khác

def main():
    # Khởi tạo GenderPredictor
    gender_predictor = GenderPredictor(rknn_model_path=MODEL_PATH)
    
    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Không thể đọc ảnh từ {image_path}")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        # Dự đoán giới tính
        gender, confidence = gender_predictor.predict_gender(image)
        frame_count += 1

        # Tính toán FPS mỗi giây
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            logging.info(f"Gender: {gender}")
            logging.info(f"Confidence: {confidence:.4f}") 
            logging.info(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

    # Đóng predictor (thực tế vòng lặp vô hạn nên dòng này không chạy, chỉ để minh họa)
    gender_predictor.close()

if __name__ == "__main__":
    main()