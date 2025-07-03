import logging
import cv2
from scipy.special import softmax
import numpy as np
from rknn.api import RKNN

ID2LABEL = {
    0: '0-2',
    1: '3-9',
    2: '10-19',
    3: '20-29',
    4: '30-39',
    5: '40-49',
    6: '50-59',
    7: '60-69',
    8: 'more than 70'
}

MODEL_PATH = "python/face_processing/models/fairface_age_float32.rknn"
# MODEL_PATH = "models/fairface_age_float32.rknn"

class AgePredictor:
    def __init__(self, rknn_model_path=MODEL_PATH, conf=0.5):
        """Initialize the age prediction model using an RKNN model."""
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
            logging.info("RKNN Age Prediction model loaded and initialized successfully.")

        except Exception as e:
            logging.error(f"Error loading RKNN model: {str(e)}")
            raise

    def predict_age(self, face_image):
        """
        Dự đoán nhóm tuổi từ ảnh khuôn mặt sử dụng mô hình RKNN.
        
        Args:
            face_image (numpy.ndarray): Ảnh khuôn mặt đầu vào (H, W, C).
            conf (float): Ngưỡng confidence tối thiểu (mặc định là 0.5). Nếu confidence nhỏ hơn conf,
                        trả về None thay vì age_range.
        
        Returns:
            tuple: (age_range, confidence) - Nhóm tuổi dự đoán và độ tin cậy.
                Trả về (None, confidence) nếu confidence < conf hoặc (None, 0.0) nếu có lỗi.
        """
        try:
            # **Tiền xử lý ảnh**
            face_image = cv2.resize(face_image, (224, 224))
            face_image = face_image.astype(np.float32)
            face_image = np.expand_dims(face_image, axis=0)
            
            # **Inference với mô hình RKNN**
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
            # logging.info(f"probs sum: {np.sum(probs)}")
            
            # **Xác định nhóm tuổi và confidence**
            age_idx = np.argmax(probs)
            confidence = probs[age_idx]
            
            # **Ánh xạ chỉ số thành nhóm tuổi**
            age_range = self.id2label.get(age_idx, "Unknown")
            
            # **Kiểm tra ngưỡng confidence**
            if confidence < self.conf:
                # logging.info(f"Confidence {confidence} < threshold {conf}, returning None")
                return None, 0.0
            
            return age_range, confidence
        
        except Exception as e:
            logging.error(f"Error during age prediction: {str(e)}")
            return None, 0.0
            
    def close(self):
        """Clean up RKNN resources.""" 
        try:
            if self.rknn:
                self.rknn.release()
                self.rknn = None
        except Exception as e:
            logging.error(f"Error releasing RKNN model: {str(e)}")
