# logging_config.py
import logging
import sys

def setup_logging():
    """
    Thiết lập cấu hình logging cho toàn bộ ứng dụng.
    - Log chỉ hiển thị trên console (stdout).
    - Định dạng log bao gồm timestamp, level, module và message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) # Ghi rõ ra stdout
        ]
    )

    # Giảm mức độ log của các thư viện bên thứ ba để tránh nhiễu
    third_party_loggers = ['PIL', 'matplotlib', 'ultralytics', 'faiss', 'rknn']
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.info("="*50)
    logging.info("Logging system initialized")
    logging.info("="*50)

def get_logger(name):
    """
    Lấy logger cho module cụ thể.
    """
    return logging.getLogger(name)