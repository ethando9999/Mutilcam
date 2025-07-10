# file: python/core/stereo_projector.py (Tối ưu hóa)

import numpy as np
import cv2
import os
import logging
import config

# Thiết lập logging
try:
    from utils.logging_python_orangepi import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

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
                 assumed_depth_mm: float = 2500.0):
        
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

    def get_robust_distance(self, rgb_box: tuple, tof_depth_map: np.ndarray) -> tuple[float | None, str]:
        """
        Ước tính khoảng cách đáng tin cậy từ một bounding box RGB bằng cách chiếu nó
        sang bản đồ độ sâu ToF và phân tích vùng ROI tương ứng.
        """
        # --- 1. Xác thực đầu vào ---
        xmin, ymin, xmax, ymax = map(int, rgb_box)
        if xmax <= xmin or ymax <= ymin:
            return None, "Box RGB không hợp lệ"
            
        # --- 2. Chiếu sơ bộ để xác định ROI trên ToF ---
        rgb_corners = np.array([[xmin, ymin], [xmax, ymax]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            tof_corners_approx = self._project_points_vectorized(rgb_corners, self.ASSUMED_DEPTH_MM)
            if tof_corners_approx is None:
                return None, "Lỗi chiếu sơ bộ"
        except Exception as e:
            logger.error(f"Lỗi trong bước chiếu sơ bộ: {e}", exc_info=True)
            return None, "Lỗi chiếu sơ bộ"
        
        # --- 3. Trích xuất ROI và lọc các điểm độ sâu hợp lệ ---
        tof_xmin, tof_ymin = np.min(tof_corners_approx, axis=0)
        tof_xmax, tof_ymax = np.max(tof_corners_approx, axis=0)
        
        h, w = tof_depth_map.shape
        tx1, ty1 = max(0, int(tof_xmin)), max(0, int(tof_ymin))
        tx2, ty2 = min(w, int(tof_xmax)), min(h, int(tof_ymax))
        
        if tx1 >= tx2 or ty1 >= ty2:
            return None, "ROI ToF rỗng"

        depth_roi = tof_depth_map[ty1:ty2, tx1:tx2]
        valid_depths = depth_roi[depth_roi > 0]
        
        # --- 4. Kiểm tra chất lượng của ROI ---
        if valid_depths.size < self.MIN_VALID_PIXELS:
            return None, f"Ít điểm D ({valid_depths.size})"
        
        # Bỏ qua các bề mặt phẳng (ví dụ: tường, sàn)
        if np.std(valid_depths) < self.MIN_STD_DEV:
            return None, f"Nhiễu phẳng (std={np.std(valid_depths):.1f})"
            
        # --- 5. Tính toán và xác thực khoảng cách cuối cùng ---
        median_depth = np.median(valid_depths)
        
        if not (self.MIN_DEPTH_MM < median_depth < self.MAX_DEPTH_MM):
            return None, f"D không hợp lệ ({median_depth:.0f})"
            
        return median_depth, "OK"

    def project_rgb_box_to_tof(self, rgb_box: tuple, tof_depth_map: np.ndarray) -> tuple | None:
        """
        Chiếu một bounding box từ RGB sang ToF để lấy box tương ứng, sử dụng khoảng cách đã được tính toán.
        Hữu ích cho việc gỡ lỗi và hiển thị.
        """
        distance_mm, status = self.get_robust_distance(rgb_box, tof_depth_map)
        if status != "OK":
            logger.warning(f"Không thể chiếu box do không lấy được khoảng cách tin cậy: {status}")
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

        # Tính toán hộp giới hạn trên ToF và đảm bảo nó nằm trong kích thước ảnh
        tof_h, tof_w = tof_depth_map.shape
        tof_xmin = max(0, int(np.min(tof_corners_projected[:, 0])))
        tof_ymin = max(0, int(np.min(tof_corners_projected[:, 1])))
        tof_xmax = min(tof_w, int(np.max(tof_corners_projected[:, 0])))
        tof_ymax = min(tof_h, int(np.max(tof_corners_projected[:, 1])))

        return (tof_xmin, tof_ymin, tof_xmax, tof_ymax)
    
        # ======================== BỔ SUNG HÀM NÀY ========================
    def get_3d_point(self, rgb_box: tuple, tof_depth_map: np.ndarray) -> np.ndarray | None:
        """
        Lấy tọa độ 3D (X, Y, Z) của tâm bounding box trong hệ tọa độ camera RGB.
        """
        depth_z, status = self.get_robust_distance(rgb_box, tof_depth_map)
        if status != "OK":
            logger.warning(f"Không thể lấy điểm 3D do lỗi khoảng cách: {status}")
            return None

        xmin, ymin, xmax, ymax = rgb_box
        center_x, center_y = (xmin + xmax) / 2., (ymin + ymax) / 2.
        
        # Unproject điểm tâm 2D về 3D
        point_x_3d = (center_x - self.cx_rgb) * depth_z / self.fx_rgb
        point_y_3d = (center_y - self.cy_rgb) * depth_z / self.fy_rgb

        return np.array([point_x_3d, point_y_3d, depth_z])