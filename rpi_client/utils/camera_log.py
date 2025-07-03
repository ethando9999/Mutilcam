import logging
import os

current_directory = os.getcwd()
# Cấu hình logging
LOG_FILE = os.path.join(current_directory, 'cam.log')

def setup_logging():
    """
    Thiết lập cấu hình logging cho toàn bộ ứng dụng rpi.
    - Log chỉ hiển thị trên console
    - Định dạng log bao gồm timestamp, level, module và message
    """
    # Cấu hình logging cơ bản 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE),  # Ghi log vào file
            # Chỉ sử dụng StreamHandler để hiển thị log trên console 
            logging.StreamHandler()
        ]
    )

    # Thiết lập log level cho một số module cụ thể
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    # Log thông tin khởi động
    logging.info("="*50)
    logging.info("Starting OrangePi logging system")
    logging.info("="*50)

def get_logger(name):
    """
    Lấy logger cho module cụ thể.
    
    Args:
        name (str): Tên của module
    
    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    return logging.getLogger(name) 