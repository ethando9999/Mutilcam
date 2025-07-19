# file: age_rknn.py

import cv2
import numpy as np
import os
import logging
from pathlib import Path

# --- Thiết lập logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from rknn.api import RKNN
except ImportError:
    logger.error("LỖI: Không tìm thấy thư viện rknn.api. Vui lòng cài đặt rknn-toolkit2.")
    raise

class AgeEstimation:
    """
    Lớp bao bọc để quản lý việc khởi tạo, chạy suy luận (inference),
    và xử lý kết quả từ mô hình phân loại độ tuổi trên nền tảng RKNN.
    """
    def __init__(self, 
                 model_path: str, 
                 class_labels: list,
                 model_width: int = 224, 
                 model_height: int = 224,
                 target_soc: str = 'rk3588'):
        self.model_path = model_path
        self.model_width = model_width
        self.model_height = model_height
        self.class_labels = class_labels

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"LỖI: Không tìm thấy model tại '{self.model_path}'")
        if not class_labels:
            raise ValueError("LỖI: Danh sách nhãn lớp (class_labels) không được để trống.")

        logger.info(f"Đang khởi tạo RKNN với model: {os.path.basename(self.model_path)}")
        self.rknn = RKNN(verbose=False)

        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            self.rknn.release()
            raise RuntimeError("LỖI: Load model RKNN thất bại.")
        
        logger.info(f"Đang khởi tạo runtime cho target: {target_soc}")
        ret = self.rknn.init_runtime(target=target_soc)
        if ret != 0:
            self.rknn.release()
            raise RuntimeError("LỖI: Khởi tạo runtime RKNN thất bại.")
            
        logger.info("✅ Khởi tạo AgeEstimation RKNN thành công.")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh đầu vào để có định dạng NHWC [1, 224, 224, 3].
        """
        # Chuyển từ BGR (OpenCV) sang RGB (model)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize ảnh về đúng kích thước đầu vào của model
        resized_img = cv2.resize(img_rgb, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)
        
        # --- SỬA LỖI TẠI ĐÂY ---
        # Chỉ thêm chiều batch (N) để tạo ra shape (1, H, W, C) -> (1, 224, 224, 3)
        input_tensor = np.expand_dims(resized_img, axis=0)
        
        return input_tensor

    def _postprocess(self, outputs: list) -> dict:
        """
        Hậu xử lý kết quả đầu ra từ model phân loại.
        """
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        predicted_index = np.argmax(probabilities)
        predicted_label = self.class_labels[predicted_index]
        confidence = probabilities[predicted_index]
        
        return {"label": predicted_label, "confidence": float(confidence)}

    def predict(self, frame: np.ndarray) -> dict:
        """
        Thực hiện dự đoán độ tuổi trên một ảnh đầu vào.
        """
        # 1. Tiền xử lý (trả về tensor shape NHWC)
        input_tensor = self._preprocess(frame)
        
        # 2. Chạy suy luận
        # --- SỬA LỖI TẠI ĐÂY ---
        # Cung cấp tensor NHWC và chỉ định rõ `data_format`
        # RKNN sẽ tự động chuyển đổi sang NCHW cho model
        outputs = self.rknn.inference(inputs=[input_tensor], data_format='nhwc')
        
        # 3. Hậu xử lý
        result = self._postprocess(outputs)
        
        return result

    def release(self):
        logger.info("Đang giải phóng tài nguyên RKNN...")
        self.rknn.release()

# --- HÀM MAIN ĐỂ CHẠY THỬ NGHIỆM TRÊN THƯ MỤC ---
if __name__ == '__main__':
    MODEL_PATH = "python/face_processing/models/age_detection_entropy.rknn"
    TEST_FOLDER_PATH = "python/quantization_rknn/archive/Data_all"
    OUTPUT_FOLDER_PATH = "python/inference_results"
    
    CLASS_LABELS = ["Adult", "Aged", "Child", "Middle Age", "Teenager"]
    
    from pathlib import Path
    import time
    from tqdm import tqdm

    age_estimator = None
    try:
        age_estimator = AgeEstimation(model_path=MODEL_PATH, class_labels=CLASS_LABELS)
        
        test_folder = Path(TEST_FOLDER_PATH)
        output_folder = Path(OUTPUT_FOLDER_PATH)
        output_folder.mkdir(parents=True, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = [p for p in test_folder.glob('*') if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            raise FileNotFoundError(f"Không tìm thấy file ảnh nào trong thư mục: {TEST_FOLDER_PATH}")
        
        logger.info(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu quá trình suy luận...")

        start_time = time.time()
        
        for image_path in tqdm(image_paths, desc="Đang xử lý ảnh"):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Bỏ qua file không thể đọc: {image_path.name}")
                continue

            result = age_estimator.predict(image)
            
            print(f"File: {image_path.name} -> Nhóm tuổi: {result['label']}, Độ tin cậy: {result['confidence']:.2%}")

            text = f"{result['label']} ({result['confidence']:.2%})"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            output_path = output_folder / image_path.name
            cv2.imwrite(str(output_path), image)

        end_time = time.time()
        total_time = end_time - start_time
        num_images = len(image_paths)
        avg_time_per_image = total_time / num_images if num_images > 0 else 0

        print("\n--- HOÀN TẤT ---")
        print(f"Đã xử lý tổng cộng: {num_images} ảnh")
        print(f"Tổng thời gian xử lý: {total_time:.2f} giây")
        print(f"Tốc độ trung bình: {avg_time_per_image * 1000:.2f} ms/ảnh")
        print(f"FPS trung bình: {1 / avg_time_per_image:.2f}")
        print(f"Ảnh kết quả đã được lưu tại thư mục: '{OUTPUT_FOLDER_PATH}'")

    except Exception as e:
        logger.error(f"Một lỗi đã xảy ra: {e}", exc_info=True)
    finally:
        if age_estimator:
            age_estimator.release()