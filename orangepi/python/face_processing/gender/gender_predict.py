import logging
import cv2
import numpy as np
from rknn.api import RKNN
from scipy.special import softmax

# Định nghĩa nhãn giới tính dựa trên mô hình từ Kaggle
ID2LABEL = {
    0: 'Male',   # Giả định 0 là nam
    1: 'Female'  # Giả định 1 là nữ
}

MODEL_PATH = "python/face_processing/models/man_woman_face_vit.rknn"
# MODEL_PATH = "models/man_woman_face_vit.rknn"

class GenderPredictor:
    def __init__(self, conf=0.5):
        """Initialize the gender prediction model using an RKNN model."""
        try:
            self.rknn = RKNN()
            logging.info(f"Loading RKNN model from: {MODEL_PATH}")
            ret = self.rknn.load_rknn(MODEL_PATH)
            if ret != 0:
                raise RuntimeError("Failed to load RKNN model")

            # Quan trọng: set target='rk3588' nếu file .rknn build cho RK3588
            ret = self.rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError("Failed to init RKNN runtime")

            self.id2label = ID2LABEL
            self.conf = conf
            logging.info("RKNN Gender Prediction model loaded and initialized successfully.")

        except Exception as e:
            logging.error(f"Error loading RKNN model: {str(e)}")
            raise

    def predict_gender(self, face_image):
        """Predict gender from a face image."""
        # Kiểm tra ảnh đầu vào
        if face_image is None or face_image.shape[0] == 0 or face_image.shape[1] == 0:
            logging.error("Invalid input image.")
            return None, 0.0

        try:
            # Resize ảnh về kích thước (224, 224) như ViT yêu cầu
            face_image = cv2.resize(face_image, (224, 224))

            # Chuyển sang định dạng float32, giữ phạm vi [0, 255]
            face_image = face_image.astype(np.float32)
            # Không cần chuẩn hóa thêm vì RKNN đã xử lý qua mean_values=[0,0,0] và std_values=[255,255,255]
            # Điều này khớp với ToTensor() trong Kaggle (chuyển [0, 255] thành [0, 1])

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
            gender_idx = np.argmax(probs)  # Lấy chỉ số có xác suất cao nhất
            gender = self.id2label.get(gender_idx, "Unknown")
            confidence = probs[gender_idx]     # Xác suất cao nhất

            # **Kiểm tra ngưỡng confidence**
            if confidence < self.conf:
                # logging.info(f"Confidence {confidence} < threshold {conf}, returning None")
                return None, 0.0
            
            return gender, confidence

        except Exception as e:
            logging.error(f"Error during gender prediction: {str(e)}")
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
    predictor = GenderPredictor(rknn_model_path="models/man_woman_face_vit.rknn")

    # Đọc ảnh mẫu (giả định)
    sample_image = cv2.imread("sample_face.jpg")
    if sample_image is not None:
        gender, confidence = predictor.predict_gender(sample_image)
        if gender:
            logging.info(f"Predicted Gender: {gender}, Confidence: {confidence:.4f}")
        else:
            logging.info("Prediction failed.")
    
    # Đóng predictor
    predictor.close()