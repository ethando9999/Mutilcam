import logging

def setup_logging():
    """
    Thiết lập cấu hình logging cho toàn bộ ứng dụng orangepi.
    - Log chỉ hiển thị trên console
    - Định dạng log bao gồm timestamp, level, module và message
    """
    # Cấu hình logging cơ bản
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # Chỉ sử dụng StreamHandler để hiển thị log trên console
            logging.StreamHandler()
        ]
    )

    # Bật INFO cho mọi logger, bao gồm __main__
    logging.getLogger().setLevel(logging.INFO)

    # Thiết lập log level cho một số module cụ thể
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('rknn').setLevel(logging.WARNING)

    # Log thông tin khởi động
    logging.info("="*50)
    logging.info("Starting OrangePi logging system")
    logging.info("="*50)

def get_logger(name): 
    """
    Lấy logger cho module cụ thể, đảm bảo level và propagate được set.
    """
    logger = logging.getLogger(name)
    # Bật level INFO và cho phép propagate lên root handler
    logger.setLevel(logging.INFO)
    logger.propagate = True
    return logger
    """
    Lấy logger cho module cụ thể.
    
    Args:
        name (str): Tên của module
    
    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    return logging.getLogger(name)
