import logging
import cv2
import numpy as np
from rknn.api import RKNN
from scipy.special import softmax

# Định nghĩa nhãn quần áo dựa trên mô hình
ID2LABEL = {
    0: 'Blazer',
    1: 'Coat',
    2: 'Denim Jacket',
    3: 'Dresses',
    4: 'Hoodie',
    5: 'Jacket',
    6: 'Jeans',
    7: 'Long Pants',
    8: 'Polo',
    9: 'Shirt',
    10: 'Shorts',
    11: 'Skirt',
    12: 'Sports Jacket',
    13: 'Sweater',
    14: 'T-shirt'
}

MODEL_PATH = "/home/ubuntu/orangepi/rknn_model_zoo-2.3.0/install/rk3588_linux_aarch64/rknn_clothes_detection_demo/model/clothes_image_detection.rknn"

class ClothesPredictor:
    def __init__(self, rknn_model_path=MODEL_PATH, conf=0.0):  # conf=0.0 để bỏ ngưỡng
        """Initialize the clothes prediction model using an RKNN model."""
        try:
            self.rknn = RKNN()
            logging.info(f"Loading RKNN model from: {rknn_model_path}")
            ret = self.rknn.load_rknn(rknn_model_path)
            if ret != 0:
                raise RuntimeError("Failed to load RKNN model")

            # Sử dụng NPU_CORE_AUTO để tối ưu FPS
            ret = self.rknn.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError("Failed to init RKNN runtime")

            self.id2label = ID2LABEL
            self.conf = conf
            logging.info("RKNN Clothes Prediction model loaded and initialized successfully.")

        except Exception as e:
            logging.error(f"Error loading RKNN model: {str(e)}")
            raise

    def predict_clothes(self, body_image):
        """Predict clothes from a body image."""
        if body_image is None or body_image.shape[0] == 0 or body_image.shape[1] == 0:
            logging.error("Invalid input image.")
            return None, 0.0

        try:
            # Log kích thước ảnh đầu vào
            logging.debug(f"Input image shape: {body_image.shape}")

            # Resize ảnh về kích thước 224x224
            body_image = cv2.resize(body_image, (224, 224))

            # Chuẩn hóa về [0, 1]
            body_image = body_image.astype(np.float32)
            body_image = body_image / 255.0

            # Chuyển BGR sang RGB
            body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)

            # Thêm batch dimension
            body_image = np.expand_dims(body_image, axis=0)  # Shape: (1, 224, 224, 3)

            # Log giá trị min/max sau preprocessing
            logging.debug(f"Preprocessed image min: {body_image.min()}, max: {body_image.max()}")

            # Suy luận với mô hình RKNN
            outputs = self.rknn.inference(inputs=[body_image], data_format='nchw')
            
            # Lấy mảng logits
            logits = outputs[0]
            if len(logits.shape) > 1:
                logits = logits[0]

            # Áp dụng softmax
            probs = softmax(logits)

            # Lấy 2 class có xác suất cao nhất
            top_indices = np.argsort(probs)[-2:][::-1]  # Lấy 2 index cao nhất, đảo ngược để giảm dần
            top_probs = probs[top_indices]
            top_labels = [self.id2label.get(idx, "Unknown") for idx in top_indices]

            # Log 2 class cao nhất với phần trăm
            logging.info(f"Top 2 predictions: {top_labels[0]} ({top_probs[0]*100:.2f}%), {top_labels[1]} ({top_probs[1]*100:.2f}%)")

            # Trả về class cao nhất và confidence
            clothes_idx = top_indices[0]
            clothes = top_labels[0]
            confidence = top_probs[0]

            # Log confidence và nhãn dự đoán
            logging.debug(f"Predicted clothes: {clothes}, Confidence: {confidence:.4f}")

            return clothes, confidence

        except Exception as e:
            logging.error(f"Error during clothes prediction: {str(e)}")
            return None, 0.0

    def close(self):
        """Clean up RKNN resources."""
        try:
            if self.rknn:
                self.rknn.release()
                self.rknn = None
                logging.info("RKNN Clothes Prediction model resources released successfully.")
        except Exception as e:
            logging.error(f"Error releasing RKNN model: {str(e)}")