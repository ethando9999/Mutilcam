# file: modules/height_estimator.py

import numpy as np
from utils.logging_python_orangepi import get_logger
from utils.kalman_filter import SimpleKalmanFilter # <-- NHẬP KHẨU

logger = get_logger(__name__)

class HeightEstimator:
    def __init__(self, camera_matrix, process_variance=0.01, measurement_variance=0.5):
        """
        Khởi tạo bộ ước tính chiều cao.

        Args:
            camera_matrix (np.array): Ma trận nội tại (mtx_rgb) 3x3 của camera.
            process_variance (float): Độ nhiễu của quá trình cho bộ lọc Kalman.
                                      Mô tả mức độ thay đổi của chiều cao thật giữa các khung hình.
                                      Vì chiều cao người không đổi, giá trị này nên nhỏ.
            measurement_variance (float): Độ nhiễu của phép đo cho bộ lọc Kalman.
                                          Mô tả mức độ tin tưởng vào phép đo chiều cao thô.
                                          Giá trị này nên lớn hơn process_variance vì phép đo bị nhiễu.
        """
        self.mtx_rgb = camera_matrix
        self.fy = self.mtx_rgb[1, 1]
        if self.fy <= 0:
            raise ValueError("Focal length (fy) must be positive.")
        
        # MỚI: Các tham số và bộ chứa cho bộ lọc Kalman
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.kalman_filters = {} # Dictionary để lưu bộ lọc cho mỗi track_id

        logger.info(f"Height Estimator initialized with fy: {self.fy:.2f} px.")
        logger.info(f"Kalman Filter params: process_var={process_variance}, measurement_var={measurement_variance}")


    def _calculate_raw_height(self, person_keypoints, distance_in_meters):
        """
        Hàm nội bộ để tính chiều cao thô (chưa qua bộ lọc).
        """
        visible_keypoints = person_keypoints[np.all(person_keypoints > 0, axis=1)]
        
        if len(visible_keypoints) < 2:
            logger.warning("Not enough visible keypoints for raw height estimation.")
            return None

        y_min = np.min(visible_keypoints[:, 1])
        y_max = np.max(visible_keypoints[:, 1])
        height_in_pixels = y_max - y_min

        if height_in_pixels <= 10:
            return None

        estimated_height_meters = (height_in_pixels * distance_in_meters) / self.fy
        
        if 0.5 < estimated_height_meters < 2.5:
            return estimated_height_meters
        else:
            logger.warning(f"Unreasonable raw height: {estimated_height_meters:.2f}m. Discarding.")
            return None

    def estimate(self, person_keypoints, distance_in_meters, track_id):
        """
        Ước tính và làm mịn chiều cao của một người bằng bộ lọc Kalman.

        Args:
            person_keypoints (np.array): Mảng keypoints (shape: 17, 2).
            distance_in_meters (float): Khoảng cách từ camera đến người.
            track_id (int or str): ID định danh duy nhất cho người đang được theo dõi.

        Returns:
            float: Chiều cao đã được làm mịn (mét), hoặc None.
        """
        # 1. Tính toán chiều cao thô từ phép đo hiện tại
        raw_height = self._calculate_raw_height(person_keypoints, distance_in_meters)
        
        if raw_height is None:
            # Nếu không có chiều cao thô, không thể cập nhật bộ lọc.
            # Trả về giá trị ước tính cuối cùng nếu có.
            if track_id in self.kalman_filters:
                return self.kalman_filters[track_id].current_estimate
            return None

        # 2. Lấy hoặc tạo bộ lọc Kalman cho track_id này
        if track_id not in self.kalman_filters:
            logger.info(f"Creating new Kalman Filter for track_id: {track_id}")
            self.kalman_filters[track_id] = SimpleKalmanFilter(
                process_variance=self.process_variance,
                measurement_variance=self.measurement_variance,
                initial_value=raw_height # Khởi tạo với giá trị đo đầu tiên
            )
        
        kalman_filter = self.kalman_filters[track_id]
        
        # 3. Cập nhật bộ lọc với phép đo mới và nhận giá trị đã làm mịn
        smoothed_height = kalman_filter.update(raw_height)
        
        logger.debug(
            f"ID-{track_id}: Raw_H={raw_height:.2f}m -> Smoothed_H={smoothed_height:.2f}m"
        )

        return smoothed_height

    def remove_tracker(self, track_id):
        """
        Xóa bộ lọc Kalman khi một đối tượng không còn được theo dõi.
        Điều này quan trọng để tránh rò rỉ bộ nhớ.
        """
        if track_id in self.kalman_filters:
            del self.kalman_filters[track_id]
            logger.info(f"Removed Kalman Filter for track_id: {track_id}")