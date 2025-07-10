import cv2
import numpy as np
import logging
import os
from config import OPI_CONFIG

# Thiết lập logger (để các lệnh logger.info, logger.error hoạt động)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

calib_file_path = OPI_CONFIG.get("calib_file_path", "depth_processing/cali_result") 


class StereoProjector:
    def __init__(self):
        """
        Khởi tạo lớp bằng cách tải các tham số hiệu chỉnh từ file .npz.
        """
        self.calib_file_path = calib_file_path
        self.params = {}
        self.load_calibration()

    def load_calibration(self): 
        """
        Tải các ma trận hiệu chỉnh từ file.
        """
        try:
            logger.info(f"Loading stereo calibration data from '{self.calib_file_path}'...")
            with np.load(self.calib_file_path) as data: 
                self.params['mtx_rgb'] = data['mtx_rgb']
                self.params['dist_rgb'] = data['dist_rgb']
                self.params['mtx_tof'] = data['mtx_tof']
                self.params['dist_tof'] = data['dist_tof']
                self.params['R'] = data['R']
                self.params['T'] = data['T']
            logger.info("Stereo calibration data loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Calibration file not found at '{self.calib_file_path}'. Cannot proceed.")
            raise
        except Exception as e:
            logger.error(f"Error loading calibration file: {e}")
            raise

    def get_depths_for_rgb_points(self, rgb_points, rgb_height, rgb_width, tof_depth_map):
        """
        Lấy giá trị chiều sâu cho một danh sách các điểm trên ảnh RGB.

        Args:
            rgb_points (list of tuples or np.array): Danh sách các tọa độ điểm (x, y) trên ảnh RGB.
                                                     Ví dụ: [(100, 150), (320, 240)].
            rgb_height (int): Chiều cao của frame ảnh RGB.
            rgb_width (int): Chiều rộng của frame ảnh RGB.
            tof_depth_map (np.array): Bản đồ chiều sâu từ camera ToF.

        Returns:
            list: Danh sách các giá trị chiều sâu (float, tính bằng mm) tương ứng với mỗi điểm đầu vào.
                  Nếu không thể tìm thấy chiều sâu cho một điểm, giá trị 0.0 sẽ được trả về cho điểm đó.
        """
        if not rgb_points:
            return []

        tof_height, tof_width = tof_depth_map.shape
        output_depths = []

        for point in rgb_points:
            # Sử dụng hàm mới để tìm tọa độ trên ảnh ToF
            tof_point = self.project_rgb_point_to_tof(point, rgb_height, rgb_width, tof_depth_map)
            
            if tof_point:
                tof_x, tof_y = tof_point
                # Lấy giá trị chiều sâu tại tọa độ đã chiếu chính xác
                final_depth = tof_depth_map[tof_y, tof_x]
                output_depths.append(float(final_depth))
            else:
                # Nếu không chiếu được điểm, trả về chiều sâu là 0
                output_depths.append(0.0)
        
        return output_depths

    def project_rgb_point_to_tof(self, rgb_point, rgb_height, rgb_width, tof_depth_map):
        """
        Chiếu một điểm từ ảnh RGB sang ảnh ToF.

        Quá trình này yêu cầu thông tin chiều sâu, được ước tính từ bản đồ chiều sâu ToF.

        Args:
            rgb_point (tuple): Tọa độ điểm (x, y) trên ảnh RGB. Ví dụ: (100, 150).
            rgb_height (int): Chiều cao của frame ảnh RGB.
            rgb_width (int): Chiều rộng của frame ảnh RGB.
            tof_depth_map (np.array): Bản đồ chiều sâu từ camera ToF.

        Returns:
            tuple: Tọa độ (x, y) tương ứng trên ảnh ToF, hoặc None nếu không thể chiếu được điểm.
        """
        tof_height, tof_width = tof_depth_map.shape
        px, py = rgb_point

        try:
            # --- BƯỚC 1: Ước tính chiều sâu (Z) bằng cách xấp xỉ vị trí trên ảnh ToF ---
            # Đây là bước quan trọng vì chúng ta cần Z để thực hiện phép chiếu chính xác.
            approx_tof_x = int(px * (tof_width / rgb_width))
            approx_tof_y = int(py * (tof_height / rgb_height))

            # Lấy chiều sâu tại điểm xấp xỉ. Lấy một vùng nhỏ để ổn định hơn.
            sample_size = 3
            tof_sample_x_start = max(0, approx_tof_x - sample_size)
            tof_sample_x_end = min(tof_width, approx_tof_x + sample_size)
            tof_sample_y_start = max(0, approx_tof_y - sample_size)
            tof_sample_y_end = min(tof_height, approx_tof_y + sample_size)

            depth_sample_region = tof_depth_map[tof_sample_y_start:tof_sample_y_end, tof_sample_x_start:tof_sample_x_end]
            valid_depths = depth_sample_region[depth_sample_region > 0]

            if len(valid_depths) == 0:
                logger.warning(f"Could not find a valid depth estimate for RGB point {rgb_point}.")
                return None

            estimated_depth = np.median(valid_depths)
            
            # --- BƯỚC 2: Chiếu chính xác điểm RGB sang ToF bằng chiều sâu đã ước tính ---
            
            # Chuẩn bị điểm RGB cho các hàm cv2
            point_rgb_2d = np.array([[rgb_point]], dtype=np.float32)

            # Hủy méo điểm RGB
            undistorted_point = cv2.undistortPoints(point_rgb_2d, self.params['mtx_rgb'], self.params['dist_rgb'], None, self.params['mtx_rgb'])
            ux, uy = undistorted_point[0][0]

            # "Unproject" từ 2D (RGB) + Z -> 3D trong hệ tọa độ camera RGB
            x_3d = (ux - self.params['mtx_rgb'][0, 2]) * estimated_depth / self.params['mtx_rgb'][0, 0]
            y_3d = (uy - self.params['mtx_rgb'][1, 2]) * estimated_depth / self.params['mtx_rgb'][1, 1]
            point_3d_rgb_cam = np.array([[x_3d, y_3d, estimated_depth]], dtype=np.float32)

            # Chiếu điểm 3D sang hệ tọa độ camera ToF
            point_2d_tof, _ = cv2.projectPoints(point_3d_rgb_cam, self.params['R'], self.params['T'], self.params['mtx_tof'], self.params['dist_tof'])

            # --- BƯỚC 3: Lấy tọa độ cuối cùng ---
            tof_x, tof_y = int(point_2d_tof[0][0][0]), int(point_2d_tof[0][0][1])

            # Kiểm tra xem tọa độ có hợp lệ không
            if 0 <= tof_x < tof_width and 0 <= tof_y < tof_height:
                return (tof_x, tof_y)
            else:
                logger.warning(f"Projected point ({tof_x}, {tof_y}) is outside ToF image bounds.")
                return None

        except Exception as e:
            logger.error(f"Error projecting RGB point {rgb_point}: {e}")
            return None
        
    def project_rgb_box_to_tof(self, rgb_box, rgb_frame, tof_depth_map):
        """
        Hàm cốt lõi: Chiếu một bounding box từ camera RGB sang camera ToF.

        Args:
            rgb_box (tuple): Bounding box trên ảnh RGB (xmin, ymin, xmax, ymax).
            rgb_frame (np.array): Frame ảnh RGB.
            tof_depth_map (np.array): Bản đồ chiều sâu từ camera ToF.

        Returns:
            tuple: Bounding box tương ứng trên ảnh ToF (xmin, ymin, xmax, ymax), hoặc None nếu thất bại.
            float: Khoảng cách trung bình của đối tượng (mm), hoặc None nếu thất bại.
        """
        # --- BƯỚC 1: Xác định vùng trung tâm trên ảnh RGB để lấy mẫu chiều sâu ---
        xmin, ymin, xmax, ymax = rgb_box
        center_x, center_y = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
        
        # Tạo một vùng nhỏ quanh trung tâm để lấy mẫu chiều sâu, tránh nhiễu ở rìa
        sample_size = 5 # Kích thước vùng lấy mẫu (5x5 pixels)
        
        # --- BƯỚC 2: Chiếu vùng trung tâm từ RGB sang ToF để tìm chiều sâu Z ---
        # Biến đổi tọa độ tâm từ RGB sang ToF (đây là một phép xấp xỉ ban đầu)
        tof_height, tof_width = tof_depth_map.shape
        rgb_height, rgb_width, _ = rgb_frame.shape
        
        approx_tof_x = int(center_x * (tof_width / rgb_width))
        approx_tof_y = int(center_y * (tof_height / rgb_height))

        # Lấy mẫu chiều sâu từ vùng xấp xỉ trên ảnh ToF
        tof_sample_x_start = max(0, approx_tof_x - sample_size)
        tof_sample_x_end = min(tof_width, approx_tof_x + sample_size)
        tof_sample_y_start = max(0, approx_tof_y - sample_size)
        tof_sample_y_end = min(tof_height, approx_tof_y + sample_size)

        depth_sample_region = tof_depth_map[tof_sample_y_start:tof_sample_y_end, tof_sample_x_start:tof_sample_x_end]
        
        # Lấy giá trị chiều sâu trung vị, loại bỏ các giá trị 0 (không hợp lệ)
        valid_depths = depth_sample_region[depth_sample_region > 0]
        if len(valid_depths) == 0:
            logger.warning("Could not find valid depth data for the target in ToF map.")
            return None, None
            
        avg_depth = np.median(valid_depths) # Dùng median để chống nhiễu tốt hơn mean
        
        # --- BƯỚC 3: Từ 2D (RGB) + Z -> 3D -> 2D (ToF) ---
        # Lấy các góc của bounding box trên ảnh RGB
        rgb_points_2d = np.array([
            [xmin, ymin], [xmax, ymin],
            [xmin, ymax], [xmax, ymax]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Hủy méo các điểm 2D này
        undistorted_rgb_points_2d = cv2.undistortPoints(rgb_points_2d, self.params['mtx_rgb'], self.params['dist_rgb'], None, self.params['mtx_rgb'])

        # Chuyển các điểm đã hủy méo thành tọa độ 3D bằng cách nhân với chiều sâu
        points_3d_rgb_cam = []
        for point in undistorted_rgb_points_2d:
            x, y = point[0]
            # Công thức "unproject" một điểm 2D về 3D
            x_3d = (x - self.params['mtx_rgb'][0, 2]) * avg_depth / self.params['mtx_rgb'][0, 0]
            y_3d = (y - self.params['mtx_rgb'][1, 2]) * avg_depth / self.params['mtx_rgb'][1, 1]
            points_3d_rgb_cam.append([x_3d, y_3d, avg_depth])
        
        points_3d_rgb_cam = np.array(points_3d_rgb_cam, dtype=np.float32).reshape(-1, 3)

        # --- BƯỚC 4: Chiếu các điểm 3D sang hệ tọa độ của camera ToF ---
        tof_points_2d, _ = cv2.projectPoints(points_3d_rgb_cam, self.params['R'], self.params['T'], self.params['mtx_tof'], self.params['dist_tof'])
        
        tof_points_2d = tof_points_2d.reshape(-1, 2)
        
        # --- BƯỚC 5: Tạo bounding box mới trên ảnh ToF ---
        tof_xmin = int(np.min(tof_points_2d[:, 0]))
        tof_xmax = int(np.max(tof_points_2d[:, 0]))
        tof_ymin = int(np.min(tof_points_2d[:, 1]))
        tof_ymax = int(np.max(tof_points_2d[:, 1]))

        # Đảm bảo bounding box không đi ra ngoài ảnh
        tof_xmin = max(0, tof_xmin)
        tof_ymin = max(0, tof_ymin)
        tof_xmax = min(tof_width, tof_xmax)
        tof_ymax = min(tof_height, tof_ymax)

        tof_box = (tof_xmin, tof_ymin, tof_xmax, tof_ymax)

        return tof_box, avg_depth

