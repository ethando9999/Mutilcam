# modules/height_estimator.py
import numpy as np
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class HeightEstimator:
    def __init__(self, camera_matrix):
        """
        Khởi tạo bộ ước tính chiều cao với ma trận nội tại của camera RGB.
        
        Args:
            camera_matrix (np.array): Ma trận nội tại (mtx_rgb) 3x3 của camera RGB.
        """
        self.mtx_rgb = camera_matrix
        # Lấy tiêu cự theo trục y (fy) từ ma trận. Vị trí (1, 1) trong ma trận.
        self.fy = self.mtx_rgb[1, 1]
        if self.fy <= 0:
            raise ValueError("Focal length (fy) must be positive.")
        logger.info(f"Height Estimator initialized with focal length (fy): {self.fy:.2f} pixels.")

    def estimate(self, person_keypoints, distance_in_meters):
        """
        Ước tính chiều cao của một người từ các keypoints và khoảng cách đã biết.

        Args:
            person_keypoints (np.array): Mảng keypoints (shape: 17, 2) cho một người.
            distance_in_meters (float): Khoảng cách từ camera đến người đó (đơn vị: mét).

        Returns:
            float: Chiều cao ước tính (mét), hoặc None nếu không đủ keypoint.
        """
        # Lọc ra các keypoint hợp lệ (có tọa độ > 0)
        visible_keypoints = person_keypoints[np.all(person_keypoints > 0, axis=1)]
        
        # Cần ít nhất 2 điểm để xác định chiều cao pixel
        if len(visible_keypoints) < 2:
            logger.warning("Not enough visible keypoints to estimate height.")
            return None

        # Tìm y_min (điểm cao nhất trên ảnh) và y_max (điểm thấp nhất trên ảnh)
        y_min = np.min(visible_keypoints[:, 1])
        y_max = np.max(visible_keypoints[:, 1])

        # Chiều cao của người trên ảnh (tính bằng pixel)
        height_in_pixels = y_max - y_min

        if height_in_pixels <= 10: # Bỏ qua nếu chiều cao pixel quá nhỏ (nhiễu)
            return None

        # Áp dụng công thức tam giác đồng dạng
        # ChieuCaoThuc = (ChieuCaoPixel * KhoangCach) / TieuCu
        estimated_height_meters = (height_in_pixels * distance_in_meters) / self.fy
        
        logger.debug(
            f"Height estimation: h_pixels={height_in_pixels}, "
            f"distance={distance_in_meters:.2f}m, fy={self.fy:.2f} -> h_real={estimated_height_meters:.2f}m"
        )
        
        # Chỉ trả về kết quả hợp lý (ví dụ: chiều cao từ 0.5m đến 2.5m)
        if 0.5 < estimated_height_meters < 2.5:
            return estimated_height_meters
        else:
            logger.warning(f"Unreasonable height calculated: {estimated_height_meters:.2f}m. Discarding.")
            return None