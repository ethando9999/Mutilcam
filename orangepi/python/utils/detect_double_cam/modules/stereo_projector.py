# file: modules/stereo_projector.py (v16 - Lấy mẫu hai giai đoạn)
import numpy as np
import cv2
from utils.logging_config import get_logger

logger = get_logger(__name__)

class StereoProjector:
    def __init__(self, calib_file_path):
        """
        Khởi tạo và tải các tham số hiệu chỉnh.
        """
        self.params = {}
        self.load_calibration(calib_file_path)

    def load_calibration(self, file_path):
        """
        Tải các ma trận hiệu chỉnh từ file .npz.
        """
        try:
            logger.info(f"Đang tải dữ liệu hiệu chỉnh stereo từ '{file_path}'...")
            with np.load(file_path) as data:
                self.params = {key: data[key] for key in data}
            logger.info("Tải dữ liệu hiệu chỉnh stereo thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi tải file hiệu chỉnh: {e}", exc_info=True)
            raise

    def _project_points(self, rgb_points_2d, depth_mm):
        """
        Hàm private chiếu một tập điểm 2D từ RGB sang ToF với một độ sâu Z cho trước.
        """
        # Hủy méo các điểm ảnh RGB
        undistorted_points = cv2.undistortPoints(
            rgb_points_2d, self.params['mtx_rgb'], self.params['dist_rgb'], None, self.params['mtx_rgb']
        )
        
        points_3d = []
        fx, fy = self.params['mtx_rgb'][0, 0], self.params['mtx_rgb'][1, 1]
        cx, cy = self.params['mtx_rgb'][0, 2], self.params['mtx_rgb'][1, 2]
        
        # Chuyển từ 2D (ảnh) sang 3D (tọa độ camera RGB)
        for point in undistorted_points:
            u, v = point[0]
            x_3d = (u - cx) * depth_mm / fx
            y_3d = (v - cy) * depth_mm / fy
            points_3d.append([x_3d, y_3d, depth_mm])
            
        # Chiếu các điểm 3D sang mặt phẳng ảnh của camera ToF
        tof_points_2d, _ = cv2.projectPoints(
            np.array(points_3d, dtype=np.float32),
            self.params['R'], self.params['T'],
            self.params['mtx_tof'], self.params['dist_tof']
        )
        return tof_points_2d.reshape(-1, 2) if tof_points_2d is not None else None

    def get_robust_distance(self, rgb_box, tof_depth_frame):
        """
        Lấy khoảng cách tin cậy bằng phương pháp "Lấy mẫu hai giai đoạn".
        """
        xmin, ymin, xmax, ymax = map(int, rgb_box)
        rgb_corners = np.array([[xmin, ymin], [xmax, ymax]], dtype=np.float32).reshape(-1, 1, 2)

        # --- GIAI ĐOẠN 1: ĐỊNH VỊ VÙNG QUAN TÂM (ROI) ---
        # Chiếu sơ bộ với một khoảng cách giả định để khoanh vùng.
        ASSUMED_DEPTH_MM = 2500.0
        try:
            tof_corners_approx = self._project_points(rgb_corners, ASSUMED_DEPTH_MM)
            if tof_corners_approx is None:
                return None, "Lỗi chiếu sơ bộ"
        except Exception as e:
            logger.error(f"Lỗi trong bước chiếu sơ bộ: {e}")
            return None, "Lỗi chiếu sơ bộ"

        tof_xmin, tof_ymin = np.min(tof_corners_approx, axis=0)
        tof_xmax, tof_ymax = np.max(tof_corners_approx, axis=0)
        
        h, w = tof_depth_frame.shape
        tx1, ty1 = max(0, int(tof_xmin)), max(0, int(tof_ymin))
        tx2, ty2 = min(w, int(tof_xmax)), min(h, int(tof_ymax))

        if tx1 >= tx2 or ty1 >= ty2:
            return None, "ROI ToF rỗng"

        # --- GIAI ĐOẠN 2: LẤY MẪU VÀ LỌC TRONG ROI ---
        depth_roi = tof_depth_frame[ty1:ty2, tx1:tx2]
        valid_depths = depth_roi[depth_roi > 0] 

        # Lọc 1: Số lượng điểm depth tối thiểu
        MIN_DEPTH_POINTS = 100
        if valid_depths.size < MIN_DEPTH_POINTS:
            return None, f"Ít điểm D ({valid_depths.size})"

        # Lọc 2: Độ lệch chuẩn (loại bỏ các vùng phẳng/lỗi)
        std_dev = np.std(valid_depths) 
        MIN_STD_DEV = 5.0
        if std_dev < MIN_STD_DEV:
            return None, f"Nhiễu phẳng (std={std_dev:.1f})"

        median_depth = np.median(valid_depths)  
        
        # Lọc 3: Loại bỏ giá trị không hợp lý  
        if not (500 < median_depth < 8000):
            return None, f"D không hợp lệ ({median_depth:.0f})" 

        logger.info(f"Lấy mẫu ROI { (tx1, ty1, tx2, ty2) }: {valid_depths.size} điểm, std={std_dev:.1f}, median={median_depth:.1f}mm")
        return median_depth, "OK"  