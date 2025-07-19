# file: python/core/stereo_projector.py (Tối ưu hóa)

import numpy as np
import cv2
import os
import logging
import math
from typing import Optional
# Thiết lập logging
try:
    from utils.logging_python_orangepi import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

class StereoProjector:
    """
    Quản lý việc chiếu điểm và hộp giới hạn giữa camera RGB và ToF.
    Phiên bản này được tối ưu hóa về tốc độ và khả năng cấu hình.
    """
    def __init__(self, calib_file_path: str,
                 min_depth_mm: int = 450,    # Điều chỉnh để chấp nhận dữ liệu giả
                 max_depth_mm: int = 8000,
                 min_valid_pixels: int = 100,
                 max_std_dev: float = 5.0,
                 assumed_depth_mm: float = 1800.0):
        
        self.params = self._load_calibration(calib_file_path)
        
        # Trích xuất các tham số camera để truy cập nhanh
        self.mtx_rgb = self.params['mtx_rgb']
        self.dist_rgb = self.params['dist_rgb']
        self.mtx_tof = self.params['mtx_tof']
        self.dist_tof = self.params['dist_tof']
        self.R = self.params['R']
        self.T = self.params['T']
        
        self.fx_rgb, self.fy_rgb = self.mtx_rgb[0, 0], self.mtx_rgb[1, 1]
        self.cx_rgb, self.cy_rgb = self.mtx_rgb[0, 2], self.mtx_rgb[1, 2]

        # Các ngưỡng cấu hình để tăng tính linh hoạt
        self.MIN_DEPTH_MM = min_depth_mm
        self.MAX_DEPTH_MM = max_depth_mm
        self.MIN_VALID_PIXELS = min_valid_pixels
        self.MIN_STD_DEV = max_std_dev # Đổi tên cho rõ nghĩa
        self.ASSUMED_DEPTH_MM = assumed_depth_mm

        logger.info("StereoProjector (tối ưu hóa) đã khởi tạo thành công.")

    def _load_calibration(self, file_path: str) -> dict:
        """Tải và xác thực file hiệu chỉnh."""
        if not os.path.exists(file_path):
            logger.error(f"File hiệu chỉnh không tồn tại tại '{file_path}'.")
            raise FileNotFoundError(f"File hiệu chỉnh không được tìm thấy tại '{file_path}'.")
        try:
            logger.info(f"Đang tải dữ liệu hiệu chỉnh stereo từ '{file_path}'...")
            with np.load(file_path) as data:
                params = {key: data[key] for key in data}  
            logger.info("Tải dữ liệu hiệu chỉnh stereo thành công.")
            return params
        except Exception as e:
            logger.error(f"Lỗi khi đọc file hiệu chỉnh '{file_path}': {e}", exc_info=True)
            raise

    def _project_points_vectorized(self, rgb_points_2d: np.ndarray, depth_mm: float) -> np.ndarray | None:
        """
        Chiếu một mảng các điểm 2D từ hệ RGB sang hệ ToF bằng phép tính vector hóa.
        """
        # 1. Undistort điểm RGB
        pts_undist = cv2.undistortPoints(rgb_points_2d, self.mtx_rgb, self.dist_rgb, P=self.mtx_rgb).reshape(-1, 2)

        # 2. Chiếu ngược về không gian 3D
        pts_3d = self._back_project_rgb_to_3d(pts_undist, depth_mm)

        # 3. Chiếu điểm 3D vào mặt phẳng ảnh ToF
        proj_2d, _ = cv2.projectPoints(pts_3d, self.R, self.T, self.mtx_tof, self.dist_tof)
        return proj_2d.reshape(-1, 2) if proj_2d is not None else None


    def _back_project_rgb_to_3d(self, pts_undist: np.ndarray, depth: float) -> np.ndarray:
        """
        Chiếu các điểm undistorted RGB 2D ra không gian 3D (hệ RGB).
        """
        u, v = pts_undist[:, 0], pts_undist[:, 1]
        x = (u - self.cx_rgb) * depth / self.fx_rgb
        y = (v - self.cy_rgb) * depth / self.fy_rgb
        z = np.full_like(x, depth)
        return np.stack((x, y, z), axis=-1)
    
    def get_robust_distance(self, rgb_box: tuple, tof_depth_map: np.ndarray, refine_steps: int = 1) -> tuple[float | None, str]:
        """
        Ước tính khoảng cách đáng tin cậy và có thể tinh chỉnh lặp lại để tăng độ chính xác.
        """
        xmin, ymin, xmax, ymax = map(int, rgb_box) 
        if xmax <= xmin or ymax <= ymin:
            return None, "Box RGB không hợp lệ"

        # Bắt đầu với khoảng cách giả định
        current_depth_mm = self.ASSUMED_DEPTH_MM
        
        # Vòng lặp tinh chỉnh (thường chỉ cần 1 lần là đủ)
        for i in range(refine_steps + 1):
            rgb_corners = np.array([[xmin, ymin], [xmax, ymax]], dtype=np.float32).reshape(-1, 1, 2)
            try:
                tof_corners = self._project_points_vectorized(rgb_corners, current_depth_mm)
                if tof_corners is None:
                    # Nếu chiếu thất bại ở bước tinh chỉnh, trả về kết quả từ bước trước đó (nếu có)
                    return (current_depth_mm, "OK") if i > 0 else (None, "Lỗi chiếu")
            except Exception as e:
                logger.warning(f"Lỗi trong bước chiếu (lần {i}): {e}")
                return (current_depth_mm, "OK") if i > 0 else (None, "Lỗi chiếu")

            # Trích xuất ROI và lọc điểm
            tof_xmin, tof_ymin = np.min(tof_corners, axis=0)
            tof_xmax, tof_ymax = np.max(tof_corners, axis=0)
            
            h, w = tof_depth_map.shape
            tx1, ty1 = max(0, int(tof_xmin)), max(0, int(tof_ymin))
            tx2, ty2 = min(w, int(tof_xmax)), min(h, int(tof_ymax))
            
            if tx1 >= tx2 or ty1 >= ty2:
                return None, "ROI ToF rỗng"

            depth_roi = tof_depth_map[ty1:ty2, tx1:tx2]
            valid_depths = depth_roi[depth_roi > 0]
            
            if valid_depths.size < self.MIN_VALID_PIXELS:
                return None, f"Ít điểm D ({valid_depths.size})"

            if np.std(valid_depths) < self.MIN_STD_DEV:
                return None, f"Nhiễu phẳng (std={np.std(valid_depths):.1f})"
            
            # Cập nhật khoảng cách ước tính cho lần lặp tiếp theo
            current_depth_mm = np.median(valid_depths)

        # Sau khi tinh chỉnh, kiểm tra lại lần cuối
        if not (self.MIN_DEPTH_MM < current_depth_mm < self.MAX_DEPTH_MM):
            return None, f"D không hợp lệ ({current_depth_mm:.0f})"
            
        return current_depth_mm, "OK"

    def project_rgb_box_to_tof(self, rgb_box: tuple, distance_mm: float, tof_depth_map_shape: tuple) -> tuple | None:
        """
        Chiếu box RGB sang ToF với một khoảng cách đã biết.
        **Lưu ý:** Hàm này không tự tính lại khoảng cách.
        """
        if distance_mm is None:
            return None

        xmin, ymin, xmax, ymax = map(int, rgb_box)
        rgb_corners = np.array([
            [xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        try:
            tof_corners_projected = self._project_points_vectorized(rgb_corners, distance_mm)
            if tof_corners_projected is None:
                return None
        except Exception as e:
            logger.error(f"Lỗi khi chiếu chính xác các góc: {e}", exc_info=True)
            return None

        tof_h, tof_w = tof_depth_map_shape
        tof_xmin = max(0, int(np.min(tof_corners_projected[:, 0])))
        tof_ymin = max(0, int(np.min(tof_corners_projected[:, 1])))
        tof_xmax = min(tof_w, int(np.max(tof_corners_projected[:, 0])))
        tof_ymax = min(tof_h, int(np.max(tof_corners_projected[:, 1])))

        return (tof_xmin, tof_ymin, tof_xmax, tof_ymax)
    
    
    # ==============================================================================
    # PHẦN VIẾT THÊM - TÍNH NĂNG MỚI ĐỂ LẤY TỌA ĐỘ THẾ GIỚI
    # ==============================================================================

    def get_worldpoint_coordinates(self,
                            torso_box: tuple,
                            feet_point_2d: Optional[tuple],
                            depth_map: np.ndarray,
                            cam_angle_deg: float) -> Optional[dict]:
        """
        Phương thức cấp cao để lấy tọa độ thế giới (sàn nhà) của một người.
        Nó kết hợp cách tiếp cận tốt nhất: lấy khoảng cách ổn định từ torso
        và chiếu từ điểm chân để có vị trí chính xác.

        Args:
            torso_box (tuple): Bounding box của thân người (xmin, ymin, xmax, ymax).
            feet_point_2d (tuple): Tọa độ (u, v) của điểm chân trên ảnh.
            depth_map (np.ndarray): Bản đồ chiều sâu từ camera ToF.
            cam_angle_deg (float): Góc nghiêng của camera so với phương ngang (độ).

        Returns:
            dict | None: Một dictionary chứa tọa độ sàn, khoảng cách và trạng thái,
                        hoặc None nếu có lỗi.
        """
        # --- BƯỚC 1: Lấy khoảng cách ổn định từ torso box ---
        distance_mm, status = self.get_robust_distance(torso_box, depth_map)
        if distance_mm is None or status != "OK":
            logger.warning(f"Không thể lấy tọa độ thế giới vì không có khoảng cách hợp lệ: {status}")
            return None

        # --- BƯỚC 2: Chiếu tọa độ sàn từ điểm chân, sử dụng khoảng cách đã có ---
        if feet_point_2d is None:
            logger.warning("Không thể lấy tọa độ thế giới vì không tìm thấy điểm chân (feet_point_2d is None).")
            # Trả về khoảng cách nhưng không có tọa độ sàn
            return {
                "floor_pos_cm": None,
                "distance_mm": distance_mm,
                "status": "Warning: No feet detected"
            }

        u_feet, v_feet = feet_point_2d

        # Chiếu ngược điểm 2D ra không gian 3D của camera
        z_cam_cm = distance_mm / 10.0
        x_cam_cm = (u_feet - self.cx_rgb) * z_cam_cm / self.fx_rgb
        y_cam_cm = (v_feet - self.cy_rgb) * z_cam_cm / self.fy_rgb

        # Bù trừ góc nghiêng của camera để có tọa độ trên mặt sàn
        angle_rad = math.radians(cam_angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        x_world = x_cam_cm
        y_world = z_cam_cm * cos_a - y_cam_cm * sin_a

        return {
            "floor_pos_cm": (x_world, y_world),
            "distance_mm": distance_mm,
            "status": status
        }