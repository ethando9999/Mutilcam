# file: modules/height_estimator.py (v20 - Lấy trung bình & Refactored)
import numpy as np
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class HeightEstimator:
    def __init__(self, mtx_rgb):
        if mtx_rgb is None or mtx_rgb.shape != (3, 3):
            raise ValueError("Ma trận nội tại của camera RGB (mtx_rgb) không hợp lệ.")
        
        self.fx, self.fy = mtx_rgb[0, 0], mtx_rgb[1, 1]
        self.cx, self.cy = mtx_rgb[0, 2], mtx_rgb[1, 2]
        
        # --- CÁC HẰNG SỐ CẤU HÌNH ---
        self.MINIMUM_DISTANCE = 0.7
        self.ANKLE_TO_FLOOR_COMPENSATION = 0.09
        self.KNEE_TO_FLOOR_COMPENSATION = 0.48
        self.TORSO_TO_HEIGHT_RATIO = 3.8
        self.HEIGHT_VALID_RANGE = (1.40, 2.10) 

        logger.info(f"HeightEstimator initialized. Method: Hierarchical 3D Projection (v20). Min_Dist: {self.MINIMUM_DISTANCE}m")

    def _project_to_3d(self, point_2d, distance_m):
        """Chiếu một điểm ảnh 2D thành điểm 3D với một khoảng cách cho trước."""
        u, v = point_2d
        X = (u - self.cx) * distance_m / self.fx
        Y = (v - self.cy) * distance_m / self.fy 
        return np.array([X, Y, distance_m])

    def _get_valid_kpts(self, keypoints, indices):
        """Lấy các keypoint hợp lệ (tọa độ > 0) từ một danh sách chỉ số."""
        return np.array([keypoints[i, :2] for i in indices if keypoints[i, 0] > 0 and keypoints[i, 1] > 0])

    def _calculate_and_validate_height(self, top_kpts, bottom_kpts, distance_m, compensation, method_name):
        """Hàm helper để tính toán, bù trừ và kiểm tra chiều cao."""
        # <<< TỐI ƯU 1: Sử dụng điểm trung bình để tăng độ ổn định >>>
        # Thay vì lấy điểm cao nhất/thấp nhất, ta lấy trung bình các điểm ở vùng đầu và vùng chân.
        top_point_2d = np.mean(top_kpts, axis=0)
        bottom_point_2d = np.mean(bottom_kpts, axis=0)
        
        point_3d_top = self._project_to_3d(top_point_2d, distance_m) 
        point_3d_bottom = self._project_to_3d(bottom_point_2d, distance_m)
        
        # Khoảng cách 3D giữa hai điểm trung bình
        height_m = np.linalg.norm(point_3d_top - point_3d_bottom) + compensation
        
        min_h, max_h = self.HEIGHT_VALID_RANGE
        if min_h < height_m < max_h:
            logger.debug(f"H tính từ '{method_name}' hợp lệ: {height_m:.2f}m")
            return height_m, method_name
        
        logger.warning(f"H từ '{method_name}' ({height_m:.2f}m) không hợp lệ.")
        return None, None

    def estimate(self, keypoints, distance_m):
        """
        Ước tính chiều cao bằng phương pháp chiếu 3D phân tầng theo bằng chứng.
        """
        if not (distance_m and self.MINIMUM_DISTANCE <= distance_m < 3.3):
            return None, f"D:Ngoài vùng ({distance_m:.1f}m)"

        # Định nghĩa các nhóm keypoint
        HEAD_KPS = [0, 1, 2, 3, 4]
        ANKLE_KPS = [15, 16]
        KNEE_KPS = [13, 14]
        SHOULDER_KPS = [5, 6]
        HIP_KPS = [11, 12]

        head_kpts = self._get_valid_kpts(keypoints, HEAD_KPS)
        
        # --- Ưu tiên 1: Dựa vào Mắt cá chân ---
        ankle_kpts = self._get_valid_kpts(keypoints, ANKLE_KPS)
        if head_kpts.size > 0 and ankle_kpts.size > 0:
            height, method = self._calculate_and_validate_height(
                head_kpts, ankle_kpts, distance_m, self.ANKLE_TO_FLOOR_COMPENSATION, "Mắt cá chân"
            )
            if height: return height, method

        # --- Ưu tiên 2: Dựa vào Đầu gối ---
        knee_kpts = self._get_valid_kpts(keypoints, KNEE_KPS)
        if head_kpts.size > 0 and knee_kpts.size > 0:
            height, method = self._calculate_and_validate_height(
                head_kpts, knee_kpts, distance_m, self.KNEE_TO_FLOOR_COMPENSATION, "Đầu gối"
            )
            if height: return height, method

        # --- Ưu tiên 3: Dựa vào Tỷ lệ thân ---
        shoulder_kpts = self._get_valid_kpts(keypoints, SHOULDER_KPS)
        hip_kpts = self._get_valid_kpts(keypoints, HIP_KPS)
        if shoulder_kpts.shape[0] == 2 and hip_kpts.shape[0] == 2:
            shoulder_mid = np.mean(shoulder_kpts, axis=0)
            hip_mid = np.mean(hip_kpts, axis=0)
            torso_len_m = np.linalg.norm(self._project_to_3d(shoulder_mid, distance_m) - self._project_to_3d(hip_mid, distance_m))
            height_m = torso_len_m * self.TORSO_TO_HEIGHT_RATIO

            min_h, max_h = self.HEIGHT_VALID_RANGE
            if min_h < height_m < max_h:
                logger.debug(f"H ước tính từ thân hợp lệ: {height_m:.2f}m")
                return height_m, "Tỷ lệ thân"
            return None, f"H Thân sai({height_m:.1f}m)"

        return None, "Ít keypoint"