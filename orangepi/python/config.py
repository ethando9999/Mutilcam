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
    "device_id": "opi",
    "db_path": os.path.join(os.path.dirname(BASE_DIR), "database", "database.db"),

    # --- Cấu hình cho Stereo Producer (put_RGBD.py) ---
    "slave_ip": "192.168.100.2",
    "tcp_port": 5005,
    "rgb_camera_id": 1,
    "rgb_device_path": "/dev/rgb_cam",
    "rgb_resolution": (640, 480),
    "rgb_framerate": 15,

    # --- Cấu hình cho Processor (processing_RGBD.py) ---
    "model_path": os.path.join(os.path.dirname(BASE_DIR), "models", "yolov8_pose.rknn"),
    "calib_file_path": os.path.join(os.path.dirname(BASE_DIR), "python/track_local/data", "calib_v2.npz"),
    "results_dir": os.path.join(os.path.dirname(BASE_DIR), "results"),
    "distance_threshold_m": 4.0, # Ngưỡng khoảng cách 4 mét

    # --- CẤU HÌNH WEBSOCKET (TÁCH BIỆT) ---
    "SOCKET_HEIGHT_URI": "ws://192.168.1.168:8080/api/ws/camera",
    "SOCKET_COUNT_URI": "ws://192.168.1.156:8080/api/ws/camera", # Giả định port và path tương tự
    "SOCKET_TABLE_ID": 1,
    # ----------------------------------------

    # --- Cấu hình Re-ID ---
    "output_dir": DEFAULT_OUTPUT_DIR,
    "feature_threshold": 0.7, "color_threshold": 0.7, "avg_threshold": 0.7,
    "top_k": 3, "thigh_weight": 8, "torso_weight": 8, "feature_weight": 0.75,
    "color_weight": 0.25, "temp_timeout": 20, "min_detections": 3,
    "merge_threshold": 0.75, "face_threshold": 0.75,
}

# --- CẤU HÌNH CHO Raspberry Pi (Single Camera) ---
RPI_CONFIG = {
    "device_id": "rpi_01", 
    "db_path": os.path.join(os.path.dirname(BASE_DIR), "database", "database_rpi.db"),
    "camera_indices": [0],
    # ... các cấu hình khác cho RPi
}

FACE_CONFIG = {
    "face_detection_conf": 0.9,
    "race_conf": 0.85,
    "gender_conf": 0.9,
    "age_conf":0.85,
}