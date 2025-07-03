import logging
import sys
import os
from datetime import datetime

def setup_logging():
    """
    Thiết lập cấu hình logging cho toàn bộ ứng dụng BananaPi.
    - Log hiển thị trên console với màu sắc
    - Log được lưu vào file theo ngày và loại (feature/frame/fps)
    - Format log bao gồm timestamp, level, module, thread và message
    - Lọc bớt log từ các thư viện bên thứ 3
    """
    class ColorFormatter(logging.Formatter):
        """Formatter tùy chỉnh để thêm màu cho log level trên console"""
        
        COLORS = {
            'DEBUG': '\033[94m',     # Blue
            'INFO': '\033[92m',      # Green  
            'WARNING': '\033[93m',   # Yellow
            'ERROR': '\033[91m',     # Red
            'CRITICAL': '\033[91m',  # Red
            'RESET': '\033[0m'       # Reset color
        }

        def format(self, record):
            # Thêm màu cho level name khi hiển thị trên console
            if record.levelname in self.COLORS:
                color_start = self.COLORS[record.levelname]
                color_end = self.COLORS['RESET']
                record.levelname = f"{color_start}{record.levelname}{color_end}"
            return super().format(record)

    # Tạo thư mục logs nếu chưa tồn tại
    log_dir = "python/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Tạo tên file log với timestamp
    current_date = datetime.now().strftime('%Y-%m-%d')
    feature_log_file = os.path.join(log_dir, f'bananapi_feature_{current_date}.log')
    frame_log_file = os.path.join(log_dir, f'bananapi_frame_{current_date}.log')
    fps_log_file = os.path.join(log_dir, f'bananapi_fps_{current_date}.log')

    # Tạo logger
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    # Format log chung
    log_format = '%(asctime)s - %(levelname)s - [%(module)s] - [%(threadName)s] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Tạo file handler cho feature log
    feature_handler = logging.FileHandler(feature_log_file) 
    feature_handler.setLevel(logging.INFO)
    feature_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    feature_handler.setFormatter(feature_formatter)
    feature_handler.addFilter(lambda record: 'feature' in record.module.lower())

    # Tạo file handler cho frame log
    frame_handler = logging.FileHandler(frame_log_file)
    frame_handler.setLevel(logging.INFO)
    frame_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    frame_handler.setFormatter(frame_formatter)
    frame_handler.addFilter(lambda record: 'frame' in record.module.lower())

    # Tạo file handler cho fps log
    fps_handler = logging.FileHandler(fps_log_file)
    fps_handler.setLevel(logging.INFO)
    fps_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    fps_handler.setFormatter(fps_formatter)
    fps_handler.addFilter(
        lambda record: ('FPS' in str(record.msg)) or 
                      ('fps' in record.module.lower())
    )

    # Tạo console handler với formatter có màu
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    color_formatter = ColorFormatter(fmt=log_format, datefmt=date_format)
    console_handler.setFormatter(color_formatter)

    # Xóa các handler cũ nếu có
    logger.handlers.clear()
    
    # Thêm tất cả handler
    logger.addHandler(feature_handler)
    logger.addHandler(frame_handler)
    logger.addHandler(fps_handler)
    logger.addHandler(console_handler)

    # Thiết lập log level cho các module bên thứ 3
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Log thông tin khởi động
    logger.info("="*50)
    logger.info("Starting BananaPi logging system")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Feature log file: {feature_log_file}")
    logger.info(f"Frame log file: {frame_log_file}")
    logger.info(f"FPS log file: {fps_log_file}")
    logger.info("="*50)

def get_logger(name):
    """
    Lấy logger cho module cụ thể.
    
    Args:
        name (str): Tên của module
    
    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    return logging.getLogger(name)

def log_fps(fps_value, module_name="unknown"):
    """
    Hàm tiện ích để log FPS với format thống nhất.
    
    Args:
        fps_value (float): Giá trị FPS cần log
        module_name (str): Tên module đang log FPS
    """
    logger = get_logger(module_name)
    logger.info(f"FPS: {fps_value:.2f}") 