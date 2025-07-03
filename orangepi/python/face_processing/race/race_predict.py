import logging
import cv2
import numpy as np
from rknn.api import RKNN
from scipy.special import softmax

# Định nghĩa nhãn cảm xúc dựa trên mô hình từ Hugging Face
ID2LABEL = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others (Hispanic, Latino, Middle Eastern)'
}
MODEL_PATH = "python/face_processing/models/model_race_utk.rknn"
# MODEL_PATH = "models/model_race_utk.rknn"

class RacePredictor:  
    def __init__(self, rknn_model_path=MODEL_PATH, conf=0.5):
        """Initialize the race prediction model using an RKNN model."""
        try:
            self.rknn = RKNN()
            logging.info(f"Loading RKNN model from: {rknn_model_path}")
            ret = self.rknn.load_rknn(rknn_model_path)
            if ret != 0:
                raise RuntimeError("Failed to load RKNN model")

            # Quan trọng: set target='rk3588' nếu file .rknn build cho RK3588
            ret = self.rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError("Failed to init RKNN runtime")

            self.id2label = ID2LABEL
            self.conf = conf
            logging.info("RKNN Race Prediction model loaded and initialized successfully.")

        except Exception as e:
            logging.error(f"Error loading RKNN model: {str(e)}")
            raise

    def predict_race(self, face_image):
        """
        Dự đoán hủng tộc từ ảnh khuôn mặt sử dụng mô hình RKNN.
        
        Args:
            face_image (numpy.ndarray): Ảnh khuôn mặt đầu vào (H, W, C).
            conf (float): Ngưỡng confidence tối thiểu (mặc định là 0.5). Nếu confidence nhỏ hơn conf,
                        trả về None thay vì race_range.
        
        Returns:
            tuple: (race_range, confidence) - chủng tộc dự đoán và độ tin cậy.
                Trả về (None, confidence) nếu confidence < conf hoặc (None, 0.0) nếu có lỗi.
        """
        # Kiểm tra ảnh đầu vào
        if face_image is None or face_image.shape[0] == 0 or face_image.shape[1] == 0:
            logging.error("Invalid input image.")
            return None, 0.0

        try:
            # Resize ảnh về kích thước (224, 224) như mô hình yêu cầu
            face_image = cv2.resize(face_image, (224, 224))

            # Chuyển sang định dạng float32, giữ phạm vi [0, 255]
            face_image = face_image.astype(np.float32)
            # Không cần chuẩn hóa thêm nếu RKNN đã xử lý qua mean_values=[0,0,0] và std_values=[255,255,255]
            # Điều này khớp với ToTensor() trong PyTorch (chuyển [0, 255] thành [0, 1])

            # Thêm batch dimension
            face_image = np.expand_dims(face_image, axis=0)  # Shape: (1, 224, 224, 3)

            # Suy luận với mô hình RKNN
            outputs = self.rknn.inference(inputs=[face_image], data_format='nhwc')
            # logging.info(f"outputs: {outputs}")
            # logging.info(f"outputs[0] shape: {outputs[0].shape}")
            
            # **Lấy mảng logits**
            logits = outputs[0]  # Mảng có shape (1, 9) hoặc (9,)
            if len(logits.shape) > 1:  # Nếu là (1, 9)
                logits = logits[0]     # Lấy hàng đầu tiên, thành (9,)
            
            # **Áp dụng softmax**
            probs = softmax(logits)
            # logging.info(f"probs after softmax: {probs}")
            # logging.info(f"probs sum: {np.sum(probs)}")  # Kiểm tra tổng xác suất có bằng 1 không

            # Xử lý kết quả đầu ra
            race_idx = np.argmax(probs)  # Lấy chỉ số có xác suất cao nhất
            race = self.id2label.get(race_idx, "Unknown")
            confidence = probs[race_idx]  # Xác suất cao nhất, chuyển sang float để dễ đọc

            # **Kiểm tra ngưỡng confidence**
            if confidence < self.conf:
                # logging.info(f"Confidence {confidence} < threshold {conf}, returning None")
                return None, 0.0

            return race, confidence

        except Exception as e:
            logging.error(f"Error during race prediction: {str(e)}")
            return None, 0.0

    def close(self):
        """Clean up RKNN resources."""
        try:
            if self.rknn:
                self.rknn.release()
                self.rknn = None
                logging.info("RKNN model resources released successfully.")
        except Exception as e:
            logging.error(f"Error releasing RKNN model: {str(e)}")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Cấu hình logging
    logging.basicConfig(level=logging.INFO)

    # Khởi tạo predictor
    predictor = RacePredictor(rknn_model_path="python/face_proccesing/models/model_race.rknn")

    # Đọc ảnh mẫu (giả định)
    sample_image = cv2.imread("orangepi/rknn/gender/4686x4000.jpg.jpg")
    if sample_image is not None:
        race, confidence = predictor.predict_race(sample_image)
        if race:
            logging.info(f"Predicted race: {race}, Confidence: {confidence:.4f}")
        else:
            logging.info("Prediction failed.")
    
    # Đóng predictor
    predictor.close()