import os

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output_frames_id"
)

veco = "vec0"
# VEC0_PATH = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "database"), veco)
VEC0_PATH = "/usr/local/lib/vec0.so"

RABBITMQ_URL = "amqp://new_user_rpi:123456@192.168.1.76:5672/"

FEATURE_DIMENSIONS = 512
FACE_DIMENSIONS = 128

# Cấu hình mặc định cho PersonReID
DEVICE_ID_CONFIG_1 = {
    "device_id": "rpi",
    "output_dir": DEFAULT_OUTPUT_DIR, 
    "feature_threshold": 0.7,  
    "color_threshold": 0.7, 
    "avg_threshold": 0.7,
    "top_k": 1,
    "thigh_weight": 8,
    "torso_weight": 8,
    "feature_weight": 0.75,
    "color_weight": 0.25,
    "db_path": "database_v1.db",
    "temp_timeout": 20, 
    "min_detections": 3,
    "merge_threshold": 0.75,
    "face_threshold": 0.75,
} 

DEVICE_ID_CONFIG_2 = {
    "device_id": "opi",
    "output_dir": DEFAULT_OUTPUT_DIR, 
    "feature_threshold": 0.7,  
    "color_threshold": 0.7, 
    "avg_threshold": 0.7,
    "top_k": 3,
    "thigh_weight": 8,
    "torso_weight": 8,
    "feature_weight": 0.75,
    "color_weight": 0.25,
    "db_path": "database_v1.db",
    "temp_timeout": 20, 
    "min_detections": 3,
    "merge_threshold": 0.75,
    "face_threshold": 0.75,
} 

FACE_CONFIG = {
    "face_detection_conf": 0.9, 
    "race_conf": 0.85,
    "gender_conf": 0.9,
    "age_conf":0.85,
}