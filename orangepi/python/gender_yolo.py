import numpy as np
from ultralytics import YOLO
from utils.logging_python_orangepi import get_logger
from typing import Optional, Dict, Any # Thêm type hinting

logger = get_logger(__name__)

class GenderRecognition:
    """
    Một class chuyên dụng để tải model YOLO phân loại giới tính
    và thực hiện dự đoán trên ảnh đầu vào với ngưỡng tin cậy.
    """
    def __init__(self, model_path: str, confidence_threshold: float = 0.75):
        """
        Khởi tạo, tải model YOLO và thiết lập ngưỡng tin cậy.
        
        Args:
            model_path (str): Đường dẫn tới file model (.pt).
            confidence_threshold (float): Ngưỡng tin cậy tối thiểu để chấp nhận dự đoán.
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            # ======================= PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =======================
            self.confidence_threshold = confidence_threshold
            # ===================== KẾT THÚC PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =====================
            logger.info(f"Đã tải thành công model GenderRecognition từ: {model_path}")
            logger.info(f"Các lớp (classes): {self.class_names}")
            logger.info(f"Ngưỡng tin cậy được thiết lập là: {self.confidence_threshold}")
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi tải model GenderRecognition: {e}", exc_info=True)
            raise e

    def predict(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Dự đoán giới tính từ một ảnh và chỉ trả về kết quả nếu
        độ tin cậy vượt qua ngưỡng đã thiết lập.
        """
        if image is None or image.size == 0:
            return None
        
        try:
            results = self.model(image, verbose=False)
            if not results or not results[0].probs:
                return None

            result = results[0]
            probs = result.probs
            
            confidence = round(probs.top1conf.item(), 2)

            # ======================= PHẦN LOGIC MỚI ĐƯỢC THÊM VÀO =======================
            # So sánh độ tin cậy với ngưỡng
            if confidence >= self.confidence_threshold:
                # Nếu đủ tin cậy, trả về kết quả dự đoán
                predicted_index = probs.top1
                predicted_label = self.class_names[predicted_index]
                
                return {
                    "gender": predicted_label,
                    "confidence": confidence
                }
            else:
                # Nếu không đủ tin cậy, trả về 'Unknown'
                # Trả về cả confidence thấp để có thể debug nếu cần
                return {
                    "gender": "Unknown",
                    "confidence": confidence
                }
            # ===================== KẾT THÚC PHẦN LOGIC MỚI ĐƯỢC THÊM VÀO =====================
            
        except Exception as e:
            logger.warning(f"Lỗi khi dự đoán giới tính: {e}", exc_info=True)
            return {"gender": "Unknown", "confidence": 0.0}