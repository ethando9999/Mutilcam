# file: python/core/height_estimator.py

import numpy as np
from utils.logging_python_orangepi import get_logger
# Sửa lại import để trỏ đến lớp trong cùng thư mục
from .stereo_projector_final import StereoProjectorFinal

logger = get_logger(__name__)

# Đổi tên lớp để khớp với lệnh import trong processing_RGBD.py
class HeightEstimatorPro: 
    def __init__(self, stereo_projector: StereoProjectorFinal):
        self.stereo_projector = stereo_projector
        
        self.MINIMUM_DISTANCE = 1.2
        self.ANKLE_TO_FLOOR_COMPENSATION = 0.08
        self.KNEE_TO_FLOOR_COMPENSATION = 0.45  
        self.TORSO_TO_HEIGHT_RATIO = 3.5
        self.HEIGHT_VALID_RANGE = (1.40, 2.10) 

        logger.info("HeightEstimator initialized. Method: Hierarchical 3D Projection.")

    def _get_3d_point_for_kpts_group(self, kpts_group: np.ndarray, tof_depth_map: np.ndarray) -> np.ndarray | None:
        if kpts_group.size == 0:
            return None
        
        xmin, ymin = np.min(kpts_group, axis=0)
        xmax, ymax = np.max(kpts_group, axis=0)
        padding = 5
        box = (xmin - padding, ymin - padding, xmax + padding, ymax + padding)
        
        return self.stereo_projector.get_3d_point(box, tof_depth_map)
        
    def _get_valid_kpts(self, keypoints, indices):
        return np.array([keypoints[i, :2] for i in indices if keypoints[i, 0] > 0 and keypoints[i, 1] > 0])

    def estimate(self, keypoints, tof_depth_map):
        # ... (Giữ nguyên toàn bộ logic của hàm estimate)
        # Định nghĩa các nhóm keypoint
        HEAD_KPS = [0, 1, 2, 3, 4]
        ANKLE_KPS = [15, 16]
        KNEE_KPS = [13, 14]
        SHOULDER_KPS = [5, 6]
        HIP_KPS = [11, 12]

        head_kpts = self._get_valid_kpts(keypoints, HEAD_KPS)
        
        point_3d_top = self._get_3d_point_for_kpts_group(head_kpts, tof_depth_map)
        if point_3d_top is None:
            return None, "Không có đầu"
            
        distance_m = point_3d_top[2] / 1000.0
        if not (self.MINIMUM_DISTANCE <= distance_m < 8.0):
            return None, f"D:Ngoài vùng ({distance_m:.1f}m)"

        ankle_kpts = self._get_valid_kpts(keypoints, ANKLE_KPS)
        point_3d_ankle = self._get_3d_point_for_kpts_group(ankle_kpts, tof_depth_map)
        if point_3d_ankle is not None:
            height_m = np.linalg.norm(point_3d_top - point_3d_ankle) + self.ANKLE_TO_FLOOR_COMPENSATION
            if self.HEIGHT_VALID_RANGE[0] < height_m < self.HEIGHT_VALID_RANGE[1]:
                return height_m, "Mắt cá chân"

        knee_kpts = self._get_valid_kpts(keypoints, KNEE_KPS)
        point_3d_knee = self._get_3d_point_for_kpts_group(knee_kpts, tof_depth_map)
        if point_3d_knee is not None:
            height_m = np.linalg.norm(point_3d_top - point_3d_knee) + self.KNEE_TO_FLOOR_COMPENSATION
            if self.HEIGHT_VALID_RANGE[0] < height_m < self.HEIGHT_VALID_RANGE[1]:
                return height_m, "Đầu gối"

        shoulder_kpts = self._get_valid_kpts(keypoints, SHOULDER_KPS)
        hip_kpts = self._get_valid_kpts(keypoints, HIP_KPS)
        point_3d_shoulder = self._get_3d_point_for_kpts_group(shoulder_kpts, tof_depth_map)
        point_3d_hip = self._get_3d_point_for_kpts_group(hip_kpts, tof_depth_map)
        if point_3d_shoulder is not None and point_3d_hip is not None:
            torso_len_m = np.linalg.norm(point_3d_shoulder - point_3d_hip)
            height_m = torso_len_m * self.TORSO_TO_HEIGHT_RATIO
            if self.HEIGHT_VALID_RANGE[0] < height_m < self.HEIGHT_VALID_RANGE[1]:
                return height_m, "Tỷ lệ thân"

        return None, "Ít keypoint"