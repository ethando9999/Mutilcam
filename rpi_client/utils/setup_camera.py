from picamera2 import Picamera2
from .camera_log import logging

FRAME_WIDTH = 640  
 
class CameraHandler:
    def __init__(self):
        """Khởi tạo CameraHandler và cấu hình camera mặc định."""
        self.picam2 = Picamera2()
        self.FRAME_HEIGHT = None 
        self.FRAME_WIDTH = FRAME_WIDTH
        self.is_default_config = False # dung de check config camera co dang o default config  khong? Neu khong thi se chuyen ve default config 
        if not self.is_default_config: 
            try:
                self.default_config_camera()  # Cấu hình camera mặc định
            except Exception as e:
                logging.error(f"Không thể thiết lập camera khi khởi tạo: {e}")

    def calculate_frame_height(self, width, height):
        """Tính toán FRAME_HEIGHT dựa trên FRAME_WIDTH và tỷ lệ khung hình."""
        if width <= 0 or height <= 0:
            raise ValueError("Chiều rộng và chiều cao phải lớn hơn 0.")
        return int(round((height / width) * self.FRAME_WIDTH))

    def default_config_camera(self): 
        """Cấu hình camera mặc định cho camera. Giam do phan giai giup giam tai cho cpu xu ly"""
        try:
            # Lấy thông tin về độ phân giải tối đa
            max_resolution = self.picam2.sensor_resolution
            if not max_resolution:
                logging.error("Camera không trả về độ phân giải hợp lệ. Vui lòng kiểm tra kết nối.")
                raise ValueError("Không thể lấy độ phân giải tối đa của camera.")

            original_width, original_height = max_resolution
            self.FRAME_HEIGHT = self.calculate_frame_height(original_width, original_height)
            reduced_resolution = (self.FRAME_WIDTH, self.FRAME_HEIGHT) 

            # Cấu hình chế độ chụp ảnh tĩnh
            still_config = self.picam2.create_still_configuration(
                main={"size": reduced_resolution}, 
                controls={"FrameRate": 30}
            )
            self.picam2.stop()  # Giải phóng camera nếu đang sử dụng
            self.picam2.configure(still_config)
            self.picam2.start()
            self.is_default_config = True  # đặt lai trạng thái camera về mặc định
            logging.info(f"Camera đã được cấu hình với độ phân giải {reduced_resolution}.")
        except Exception as e:
            logging.error(f"Lỗi cấu hình camera: {e}") 

    def change_max_resolution(self):
        """Chuyển sang cấu hình max cho camera khi phát hiện chuyển động và tính toán tỷ lệ scale."""
        try:
            max_resolution = self.picam2.sensor_resolution
            max_width, max_height = max_resolution

            still_config = self.picam2.create_still_configuration(
                main={"format": "RGB888", "size": (max_resolution)},
                controls={"FrameRate": 10}
            )
            # Dừng camera, áp dụng cấu hình mới, sau đó khởi động lại
            self.picam2.stop()
            self.picam2.configure(still_config)
            self.picam2.start()
            self.is_default_config = False # Da doi config
            logging.info(f"Resolution changed to {max_resolution} successfully.")

            # Tính toán tỷ lệ scale cho x và y
            if not self.FRAME_HEIGHT:
                logging.error("FRAME_HEIGHT chưa được thiết lập. Hãy gọi setup_camera trước.")
                raise ValueError("FRAME_HEIGHT is not initialized.")

            x_scale = self.FRAME_WIDTH / max_width
            y_scale = self.FRAME_HEIGHT / max_height
            return x_scale, y_scale
        except Exception as e:
            logging.error(f"Failed to change camera resolution: {e}")
            return None, None

    def stop_camera(self):
        """Dừng camera khi không sử dụng nữa."""
        try:
            self.picam2.stop()
            logging.info("Camera đã được dừng.")
        except Exception as e:
            logging.error(f"Lỗi khi dừng camera: {e}")

    def __del__(self):
        """Giải phóng camera khi đối tượng bị hủy."""
        try:
            self.stop_camera()
        except Exception as e:
            logging.warning(f"Lỗi khi giải phóng camera trong destructor: {e}")

    def capture_image(self):
        """
        Chụp ảnh trực tiếp từ camera sử dụng capture_array.
        
        Returns:
            numpy.ndarray: Ảnh được chụp hoặc None nếu có lỗi
        """
        try:
            # Chụp ảnh từ camera
            frame = self.picam2.capture_array()
            
            # Log thông tin về frame
            if frame is not None:
                height, width = frame.shape[:2]
                size_kb = frame.nbytes / 1024
                logging.debug(f"Captured frame size: {width}x{height}, {size_kb:.2f}KB")
                return frame
            else:
                logging.error("Capture_array returned None")
                return None
            
        except Exception as e:
            logging.error(f"Lỗi khi chụp ảnh: {e}", exc_info=True)
            return None
