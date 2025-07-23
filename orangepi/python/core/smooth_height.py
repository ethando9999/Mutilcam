# file: utils/smoothing.py
import numpy as np
from typing import List, Dict, Any, Optional
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Lớp SmoothHeightFilter không thay đổi
class SmoothHeightFilter:
    """
    Bộ lọc Kalman 1D đơn giản để làm mịn giá trị theo thời gian cho một đối tượng duy nhất.
    """
    def __init__(self, process_variance, measurement_variance, estimated_error=1.0, initial_value=None):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_error = estimated_error
        self.current_estimate = initial_value
        
    def update(self, measurement):
        if self.current_estimate is None:
            self.current_estimate = measurement
            return self.current_estimate
        prediction_error = self.estimated_error + self.process_variance
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.current_estimate = self.current_estimate + kalman_gain * (measurement - self.current_estimate)
        self.estimated_error = (1 - kalman_gain) * prediction_error
        return self.current_estimate

class HeightResultProcessor:
    """
    Xử lý, làm mịn và xác thực các kết quả chiều cao trước khi gửi đi.
    Lớp này bây giờ quản lý trạng thái để xử lý các thay đổi đột ngột.
    """
    def __init__(self):
        """
        Khởi tạo bộ xử lý với các biến quản lý trạng thái.
        """
        logger.info("HeightResultProcessor (có logic xác thực) đã được khởi tạo.")
        
        # --- CÁC HẰNG SỐ CẤU HÌNH CHO VIỆC XÁC THỰC ---
        self.VALIDATION_THRESHOLD_CM = 20.0  # Ngưỡng chênh lệch chiều cao để kích hoạt xác thực
        self.VALIDATION_FRAMES = 5          # Số frame cần để xác thực giá trị mới
        self.STABILITY_MARGIN_CM = 20.0     # Biên độ cho phép để coi là "ổn định"

        # --- CÁC BIẾN TRẠNG THÁI ---
        self.last_sent_height_cm: Optional[float] = None
        self.validation_pending: bool = False
        self.validation_candidate_height_cm: Optional[float] = None
        self.validation_counter: int = 0

        # Placeholder cho tính năng làm mịn theo từng người trong tương lai
        self.individual_filters: Dict[int, SmoothHeightFilter] = {}
        
    def process_and_validate_heights(self, raw_heights_cm: List[float]) -> Optional[List[float]]:
        """
        Phương thức chính: xử lý, làm mịn và xác thực chiều cao.
        Trả về một mảng chiều cao nếu cần gửi đi, ngược lại trả về None.
        """
        if not raw_heights_cm:
            return None # Không có dữ liệu, không gửi gì
        
        if raw_heights_cm == 0.0: 
            return 0.0

        current_min_height = min(raw_heights_cm)
        num_people = len(raw_heights_cm)

        # --- Luồng 1: Đang trong quá trình xác thực ---
        if self.validation_pending:
            # Kiểm tra xem chiều cao hiện tại có "ổn định" so với giá trị đang xác thực không
            if abs(current_min_height - self.validation_candidate_height_cm) <= self.STABILITY_MARGIN_CM:
                self.validation_counter += 1
                logger.debug(f"Xác thực chiều cao mới ({self.validation_candidate_height_cm:.1f}cm). கவுন্টার: {self.validation_counter}/{self.VALIDATION_FRAMES}")
                
                # Nếu đủ số frame xác thực
                if self.validation_counter >= self.VALIDATION_FRAMES:
                    logger.info(f"✅ Xác thực thành công! Chấp nhận chiều cao mới: {self.validation_candidate_height_cm:.1f}cm")
                    self.last_sent_height_cm = self.validation_candidate_height_cm
                    # Reset trạng thái
                    self.validation_pending = False
                    self.validation_counter = 0
                    self.validation_candidate_height_cm = None
                    return [self.last_sent_height_cm] * num_people
            else:
                # Chiều cao không ổn định (giảm đột ngột), hủy bỏ xác thực
                logger.warning(f"❌ Xác thực thất bại. Chiều cao giảm xuống {current_min_height:.1f}cm. Hủy bỏ ứng viên {self.validation_candidate_height_cm:.1f}cm.")
                self.validation_pending = False
                self.validation_counter = 0
                self.validation_candidate_height_cm = None
                # Cập nhật và gửi đi giá trị thấp hiện tại luôn
                self.last_sent_height_cm = current_min_height
                return [self.last_sent_height_cm] * num_people
            
            return None # Vẫn đang trong quá trình xác thực, không gửi gì

        # --- Luồng 2: Trạng thái bình thường (không xác thực) ---
        # Trường hợp đầu tiên, chưa có giá trị nào được gửi
        if self.last_sent_height_cm is None:
            self.last_sent_height_cm = current_min_height
            logger.info(f"Nhận chiều cao lần đầu: {self.last_sent_height_cm:.1f}cm. Sẵn sàng gửi đi.")
            return [self.last_sent_height_cm] * num_people

        # Kiểm tra nếu có sự tăng vọt về chiều cao
        if current_min_height > self.last_sent_height_cm + self.VALIDATION_THRESHOLD_CM:
            logger.warning(f"Phát hiện chiều cao tăng đột ngột: {self.last_sent_height_cm:.1f}cm -> {current_min_height:.1f}cm. Bắt đầu quá trình xác thực...")
            self.validation_pending = True
            self.validation_candidate_height_cm = current_min_height
            self.validation_counter = 1 # Bắt đầu đếm từ frame này
            return None # Không gửi gói tin, chờ xác thực
        else:
            # Cập nhật bình thường (chiều cao giảm hoặc tăng nhẹ)
            self.last_sent_height_cm = current_min_height
            return [self.last_sent_height_cm] * num_people