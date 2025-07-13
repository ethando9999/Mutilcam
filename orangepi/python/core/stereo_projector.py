# file: python/core/stereo_projector.py (Tối ưu hóa)

import numpy as np
import cv2
import os
import logging

# Thiết lập logging
try:
    from utils.logging_python_orangepi import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
import config
calib_file_path = config.OPI_CONFIG.get("calib_file_path", "depth_processing/cali_result")

class StereoProjector:
    """
    Quản lý việc chiếu điểm và hộp giới hạn giữa camera RGB và ToF.
    Phiên bản này được tối ưu hóa về tốc độ và khả năng cấu hình.
    """
    def __init__(self,
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
        Nhanh hơn đáng kể so với việc lặp qua từng điểm.
        """
        # 1. Undistort điểm RGB
        undistorted_points = cv2.undistortPoints(rgb_points_2d, self.mtx_rgb, self.dist_rgb, None, self.mtx_rgb)
        
        # 2. Chiếu điểm 2D undistorted ra 3D
        u_v = undistorted_points.reshape(-1, 2)
        x_3d = (u_v[:, 0] - self.cx_rgb) * depth_mm / self.fx_rgb
        y_3d = (u_v[:, 1] - self.cy_rgb) * depth_mm / self.fy_rgb
        
        points_3d = np.stack([x_3d, y_3d, np.full_like(x_3d, depth_mm)], axis=-1)

        # 3. Chiếu điểm 3D vào mặt phẳng ảnh ToF
        tof_points_2d, _ = cv2.projectPoints(points_3d, self.R, self.T, self.mtx_tof, self.dist_tof)
        
        return tof_points_2d.reshape(-1, 2) if tof_points_2d is not None else None

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
