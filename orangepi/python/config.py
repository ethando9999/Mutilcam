# file: python/config.py

import os

# Đường dẫn gốc của dự án, giả sử file config.py nằm trong thư mục python/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# <<< ======================= THÊM CÁC HẰNG SỐ BỊ THIẾU ======================= >>>
FEATURE_DIMENSIONS = 512    # Kích thước vector đặc trưng của OSNet
FACE_DIMENSIONS = 128     # Kích thước vector đặc trưng của MobileFaceNet
# <<< ===================== KẾT THÚC THÊM HẰNG SỐ ===================== >>>

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(BASE_DIR), # Đi lên một cấp để ra thư mục orangepi/
    "results",
    "output_frames_id"
)
VEC0_PATH = "/usr/local/lib/vec0.so"
RABBITMQ_URL = "amqp://new_user_rpi:123456@192.168.1.76:5672/"

# --- CẤU HÌNH CHO Orange Pi (Stereo Vision) ---
OPI_CONFIG = {
    # --- Định danh và DB ---
    "device_id": "opi_01",
    "db_path": os.path.join(os.path.dirname(BASE_DIR), "database", "database_opi.db"),

    # --- Cấu hình Producer ---
    "slave_ip": "192.168.100.2", 
    "tcp_port": 5005,
    "rgb_camera_id": 0, 
    "rgb_device_path": "/dev/rgb_cam",
    "rgb_resolution": (640, 480),
    "rgb_framerate": 15,

    # --- Cấu hình Processor ---
    "calib_file_path": os.path.join(os.path.dirname(BASE_DIR), "python/calibration/cali_result", "cam105_0353.npz"),
    "results_dir": os.path.join(os.path.dirname(BASE_DIR), "results"),
    "distance_threshold_m": 4.0,

    ### <<< THÊM CẤU HÌNH WEBSOCKET VÀO ĐÂY >>> ###
    "SOCKET_COUNT_URI": "ws://192.168.1.43:8080/api/ws/camera",
    "SOCKET_HEIGHT_URI": "ws://192.168.1.168:8080/api/ws/camera",
    "SOCKET_ID_URI": "ws://192.168.1.43:9090/api/ws/camera",
    "SOCKET_TABLE_ID": 1,

    # --- Cấu hình Re-ID ---
    "output_dir": DEFAULT_OUTPUT_DIR,
    "feature_threshold": 0.5, "color_threshold": 0.7, "face_threshold": 0.5,
    "hard_feature_threshold": 0.62, "hard_face_threshold": 0.7, "top_k": 3,  
    "merge_threshold": 0.75, "temp_timeout": 20, "min_detections": 3,
}


# --- CẤU HÌNH CHO Raspberry Pi (Single Camera) --- 
RPI_CONFIG = {
    "device_id": "rpi_01",
    "db_path": os.path.join(os.path.dirname(BASE_DIR), "database", "database_rpi.db"), 
    "camera_indices": [0],
    # ... các cấu hình khác cho RPi
} 

FACE_CONFIG = {
    "face_detection_conf": 0.8,  
    "race_conf": 0.7,
    "gender_conf": 0.9, 
    "age_conf": 0.8,
}