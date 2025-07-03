from picamera2 import Picamera2
from utils.logging_python_orangepi import get_logger
import subprocess
import time

logger = get_logger(__name__)

FRAME_WIDTH_cam_0 = 3000 # Cập nhật giá trị chuẩn 
FRAME_WIDTH_cam_1 = 2592

class CameraHandler:
    def __init__(self, camera_index=0):
        """
        Khởi tạo CameraHandler và cấu hình camera mặc định.
        :param camera_index: Chọn camera (0 cho /dev/video0, 1 cho /dev/video1)
        """
        try:
            self.camera_index = camera_index
            self.device_path = f"/dev/video{self.camera_index}"
            
            # Giải phóng camera nếu bị chiếm dụng
            self._release_busy_camera()
            
            self.picam2 = Picamera2(camera_num=self.camera_index)
            self.FRAME_HEIGHT = None
            
            # Cập nhật FRAME_WIDTH tương ứng với camera_index
            if self.camera_index == 0:
                self.FRAME_WIDTH = FRAME_WIDTH_cam_0
            elif self.camera_index == 1:
                self.FRAME_WIDTH = FRAME_WIDTH_cam_1
            else:
                raise ValueError("camera_index không hợp lệ.")
                
            self.x_scale = None
            self.y_scale = None
            self.is_default_config = False
            self.default_config_camera()
        except Exception as e:
            logger.error(f"Không thể thiết lập camera {self.camera_index}: {e}")

    def _release_busy_camera(self):
        """Giải phóng camera nếu đang bị chiếm dụng."""
        try:
            cmd = f"sudo fuser -k {self.device_path}"
            subprocess.run(cmd.split(), check=False)
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Lỗi khi giải phóng camera {self.camera_index}: {e}")

    def calculate_frame_height(self, width, height):
        """Tính toán FRAME_HEIGHT dựa trên FRAME_WIDTH."""
        if width <= 0 or height <= 0:
            raise ValueError("Chiều rộng và chiều cao phải lớn hơn 0.")
        return int(round((height / width) * self.FRAME_WIDTH))

    def calculate_scale_factors(self, original_width, original_height):
        """Tính toán x_scale và y_scale."""
        if original_width <= 0 or original_height <= 0:
            raise ValueError("Chiều rộng và chiều cao gốc phải lớn hơn 0.")
        self.x_scale = self.FRAME_WIDTH / original_width
        self.y_scale = self.FRAME_HEIGHT / original_height

    def default_config_camera(self):
        """Cấu hình camera mặc định."""
        try:
            max_resolution = self.picam2.sensor_resolution
            if not max_resolution:
                raise ValueError("Không thể lấy độ phân giải tối đa của camera.")
            
            original_width, original_height = max_resolution
            self.FRAME_HEIGHT = self.calculate_frame_height(original_width, original_height)
            self.calculate_scale_factors(original_width, original_height)
            reduced_resolution = (self.FRAME_WIDTH, self.FRAME_HEIGHT)
            
            config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": reduced_resolution},
                lores={"size": reduced_resolution},
                controls={"FrameRate": 20}
            )
            
            self.picam2.stop()
            self.picam2.configure(config)
            self.picam2.start()
            self.is_default_config = True
            logger.info(f"Camera {self.camera_index} cấu hình {reduced_resolution} thành công.")
        except Exception as e:
            logger.error(f"Lỗi cấu hình camera {self.camera_index}: {e}")

    def capture_lores_frame(self):
        try:
            return self.picam2.capture_array("lores")
        except Exception as e:
            logger.error(f"Lỗi khi chụp ảnh lores camera {self.camera_index}: {e}")
            return None

    def capture_main_frame(self):
        try:
            return self.picam2.capture_array("main")
        except Exception as e:
            logger.error(f"Lỗi khi chụp ảnh main camera {self.camera_index}: {e}")
            return None

    def stop_camera(self):
        try:
            self.picam2.stop()
            logger.info(f"Camera {self.camera_index} đã dừng.")
        except Exception as e:
            logger.error(f"Lỗi khi dừng camera {self.camera_index}: {e}")

    def __del__(self):
        self.stop_camera()

# # Sử dụng 2 camera
# camera0 = CameraHandler(camera_index=0)
# camera1 = CameraHandler(camera_index=1)
