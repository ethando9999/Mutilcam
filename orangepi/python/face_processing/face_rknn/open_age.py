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
        """
        Khởi tạo và tải model RKNN.

        Args:
            model_path (str): Đường dẫn đến file .rknn.
            class_labels (list): Danh sách các nhãn lớp theo đúng thứ tự model được huấn luyện.
            model_width (int): Chiều rộng đầu vào của model.
            model_height (int): Chiều cao đầu vào của model.
            target_soc (str): Tên SoC mục tiêu (ví dụ: 'rk3588').
        """
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

        # Tải model
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            self.rknn.release()
            raise RuntimeError("LỖI: Load model RKNN thất bại.")
        
        # Khởi tạo runtime
        logger.info(f"Đang khởi tạo runtime cho target: {target_soc}")
        ret = self.rknn.init_runtime(target=target_soc)
        if ret != 0:
            self.rknn.release()
            raise RuntimeError("LỖI: Khởi tạo runtime RKNN thất bại.")
            
        logger.info("✅ Khởi tạo AgeEstimation RKNN thành công.")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh đầu vào: Chuyển đổi màu sắc và resize.
        Mô hình SigLIP được huấn luyện với ảnh RGB.
        Việc chuẩn hóa (normalize) đã được cấu hình trong rknn.config() 
        nên runtime sẽ tự động thực hiện.
        """
        # Chuyển từ BGR (OpenCV) sang RGB (model)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize ảnh về đúng kích thước đầu vào của model
        resized_img = cv2.resize(img_rgb, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)
        
        return resized_img

    def _postprocess(self, outputs: list) -> dict:
        """
        Hậu xử lý kết quả đầu ra từ model phân loại.
        Hàm này tìm ra lớp có xác suất cao nhất và trả về nhãn cùng độ tin cậy.
        """
        # Lấy tensor đầu ra (logits) từ model
        # outputs là một list, ta lấy phần tử đầu tiên
        logits = outputs[0][0] # Shape (1, num_classes) -> (num_classes,)
        
        # Áp dụng hàm softmax để chuyển logits thành xác suất
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        # Tìm chỉ số của lớp có xác suất cao nhất
        predicted_index = np.argmax(probabilities)
        
        # Lấy nhãn và độ tin cậy tương ứng
        predicted_label = self.class_labels[predicted_index]
        confidence = probabilities[predicted_index]
        
        return {"label": predicted_label, "confidence": float(confidence)}

    def predict(self, frame: np.ndarray) -> dict:
        """
        Thực hiện dự đoán độ tuổi trên một ảnh đầu vào.

        Args:
            frame (np.ndarray): Ảnh BGR để dự đoán.

        Returns:
            dict: Một dictionary chứa kết quả dự đoán.
                  Ví dụ: {'label': 'Adult', 'confidence': 0.95}.
        """
        # 1. Tiền xử lý
        input_image = self._preprocess(frame)
        
        # 2. Chạy suy luận
        # Cung cấp ảnh đã xử lý dưới dạng list
        outputs = self.rknn.inference(inputs=[input_image])
        
        # 3. Hậu xử lý
        result = self._postprocess(outputs)
        
        return result

    def release(self):
        """
        Giải phóng tài nguyên RKNN.
        """
        logger.info("Đang giải phóng tài nguyên RKNN...")
        self.rknn.release()

# --- HÀM MAIN ĐỂ CHẠY THỬ NGHIỆM ---
if __name__ == '__main__':
    # --- Cấu hình ---
    MODEL_PATH = "orangepi/python/face_processing/models/age_detection_entropy.rknn"
    # Đường dẫn đến thư mục chứa các ảnh cần thử nghiệm
    TEST_FOLDER_PATH = "orangepi/python/quantization_rknn/archive/Data_all"
    # Thư mục để lưu các ảnh kết quả
    OUTPUT_FOLDER_PATH = "inference_results"
    
    # RẤT QUAN TRỌNG: Danh sách các nhãn lớp
    CLASS_LABELS = [
        "Adult", 
        "Aged", 
        "Child",
        "Middle Age", 
        "Teenager"
    ]
    
    # Thêm import cần thiết
    from pathlib import Path
    import time
    from tqdm import tqdm

    # Khởi tạo đối tượng
    age_estimator = None
    try:
        # --- Bước 1: Khởi tạo model ---
        age_estimator = AgeEstimation(model_path=MODEL_PATH, class_labels=CLASS_LABELS)
        
        # --- Bước 2: Chuẩn bị danh sách ảnh và thư mục output ---
        test_folder = Path(TEST_FOLDER_PATH)
        output_folder = Path(OUTPUT_FOLDER_PATH)
        output_folder.mkdir(parents=True, exist_ok=True) # Tạo thư mục output nếu chưa có

        # Lấy danh sách tất cả các file ảnh
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = [p for p in test_folder.glob('*') if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            raise FileNotFoundError(f"Không tìm thấy file ảnh nào trong thư mục: {TEST_FOLDER_PATH}")
        
        logger.info(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu quá trình suy luận...")

        # --- Bước 3: Chạy suy luận trên từng ảnh ---
        start_time = time.time()
        
        for image_path in tqdm(image_paths, desc="Đang xử lý ảnh"):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Bỏ qua file không thể đọc: {image_path.name}")
                continue

            # Thực hiện dự đoán
            result = age_estimator.predict(image)
            
            # In kết quả của file hiện tại
            print(f"File: {image_path.name} -> Nhóm tuổi: {result['label']}, Độ tin cậy: {result['confidence']:.2%}")

            # Vẽ kết quả lên ảnh
            text = f"{result['label']} ({result['confidence']:.2%})"
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Lưu ảnh kết quả vào thư mục output
            output_path = output_folder / image_path.name
            cv2.imwrite(str(output_path), image)

        # --- Bước 4: Thống kê hiệu năng ---
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
        logger.error(f"Một lỗi đã xảy ra: {e}")
    finally:
        # Luôn giải phóng tài nguyên dù có lỗi hay không
        if age_estimator:
            age_estimator.release()