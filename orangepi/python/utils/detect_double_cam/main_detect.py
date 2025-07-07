# file: main_app.py (v17 - Lưu ảnh crop & Công thức mới)
import cv2
import numpy as np
import time
import os
import socket
import json
import struct
import threading
from datetime import datetime

from orangepi.python.utils.yolo_pose_rknn import HumanDetection
from modules.stereo_projector import StereoProjector
from modules.height_estimator import HeightEstimator
from utils.kalman_filter import SimpleKalmanFilter
from utils.logging_config import setup_logging, get_logger

# --- CẤU HÌNH ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
setup_logging(log_dir=LOG_DIR, log_file="main_app.log")
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE_PATH = os.path.join(BASE_DIR, "data", "stereo_calib_result.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "results") 
ANNOTATED_RGB_DIR = os.path.join(RESULTS_DIR, "annotated_rgb")
# <<< TỐI ƯU: THÊM ĐƯỜNG DẪN LƯU ẢNH CROP >>>
RGB_CROPS_DIR = os.path.join(RESULTS_DIR, "rgb_crops")


RGB_CAMERA_ID = 0; SLAVE_IP = "192.168.100.2"; TCP_PORT = 5005
latest_rgb_frame = None; lock = threading.Lock(); stop_event = threading.Event()

# --- CÁC HÀM PHỤ TRỢ (Giữ nguyên) ---
def frame_reader(cap): # ...
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            with lock: global latest_rgb_frame; latest_rgb_frame = frame.copy()
        else: time.sleep(0.01)

def connect_to_slave(): # ...
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(10.0)
        s.connect((SLAVE_IP, TCP_PORT)); logger.info(">>> Kết nối Slave thành công! <<<")
        return s
    except Exception as e:
        logger.error(f"Kết nối Slave thất bại: {e}"); return None

def recv_all(sock, num_bytes): # ...
    buffer = bytearray()
    while len(buffer) < num_bytes:
        packet = sock.recv(num_bytes - len(buffer))
        if not packet: raise ConnectionError(f"Kết nối đã đóng, chỉ nhận được {len(buffer)}/{num_bytes} bytes.")
        buffer.extend(packet)
    return buffer  

def request_and_receive_tof_frames(sock):
    """Gửi lệnh và nhận dữ liệu frame từ Slave."""
    try:
        sock.sendall(b"CAPTURE")
        header_len_bytes = recv_all(sock, 4)
        if not header_len_bytes: return None, None
        header_len = struct.unpack('>I', header_len_bytes)[0]
        if header_len == 0:
            logger.error("Received an error signal (empty header) from Slave.")
            return None, None
        header_bytes = recv_all(sock, header_len)
        if not header_bytes: return None, None
        header = json.loads(header_bytes.decode('utf-8'))
        depth_size = header['depth_shape'][0] * header['depth_shape'][1] * np.dtype(header['depth_dtype']).itemsize
        depth_bytes = recv_all(sock, depth_size)
        if not depth_bytes: return None, None
        depth_frame = np.frombuffer(depth_bytes, dtype=np.dtype(header['depth_dtype'])).reshape(header['depth_shape'])
        amp_size = header['amp_shape'][0] * header['amp_shape'][1] * np.dtype(header['amp_dtype']).itemsize
        amp_bytes = recv_all(sock, amp_size)
        if not amp_bytes: return None, None
        amp_frame = np.frombuffer(amp_bytes, dtype=np.dtype(header['amp_dtype'])).reshape(header['amp_shape'])
        return depth_frame, amp_frame
    except Exception as e:
        logger.error(f"Error during data reception: {e}")
        return None, None

def process_person(person_idx, rgb_box, kpts, projector, tof_frame, height_estimator, kalman_filters):
    # (Hàm này giữ nguyên)
    person_id_str = f"P{person_idx+1}"
    dist_mm, dist_status = projector.get_robust_distance(rgb_box, tof_frame)
    if dist_status != "OK":
        logger.warning(f"❌ {person_id_str} - Lỗi khoảng cách: {dist_status}")
        return {'text': f"{person_id_str} [{dist_status}]", 'color': (0, 165, 255)}
    if not dist_mm or dist_mm > 8000: return {'text': f"{person_id_str} [Ngoài tầm]", 'color': (0, 0, 255)}
    distance_m = dist_mm / 1000.0
    est_height, height_status = height_estimator.estimate(kpts, distance_m)
    if not est_height:
        logger.warning(f"❌ {person_id_str} - Lỗi chiều cao: {height_status} (D={distance_m:.1f}m)")
        return {'text': f"{person_id_str} D:{distance_m:.1f}m [{height_status}]", 'color': (0, 0, 255)}
    logger.info(f"✅ {person_id_str} - Chiều cao thô ({height_status}): {est_height:.2f}m")
    if person_idx not in kalman_filters: kalman_filters[person_idx] = SimpleKalmanFilter(1e-4, 0.05)
    smooth_height = kalman_filters[person_idx].update(est_height)
    logger.info(f"✅ {person_id_str} - Chiều cao sau Kalman: {smooth_height:.2f}m")
    return {'text': f"{person_id_str} D:{distance_m:.2f}m H:{smooth_height:.2f}m", 'color': (0, 255, 0)}

def main():
    logger.info("="*50 + "\nKhởi tạo ứng dụng đo chiều cao (v17 - Công thức mới)\n" + "="*50)
    # <<< TỐI ƯU: TẠO TẤT CẢ CÁC THƯ MỤC CẦN THIẾT >>>
    for path in [ANNOTATED_RGB_DIR, RGB_CROPS_DIR]: os.makedirs(path, exist_ok=True)
    
    try:
        detector, projector = HumanDetection(), StereoProjector(CALIB_FILE_PATH)
        height_estimator, kalman_filters = HeightEstimator(projector.params['mtx_rgb']), {}
    except Exception as e: logger.error(f"Lỗi khởi tạo: {e}", exc_info=True); return

    cap_rgb = cv2.VideoCapture(RGB_CAMERA_ID)
    if not cap_rgb.isOpened(): logger.error(f"Lỗi camera RGB ID {RGB_CAMERA_ID}"); return
    
    reader_thread = threading.Thread(target=frame_reader, args=(cap_rgb,), daemon=True); reader_thread.start()
    time.sleep(2)

    slave_socket = None
    try:
        while True:
            if input("\n>>> Nhấn ENTER để xử lý, 'q' để thoát: ").lower() == 'q': break
            
            if slave_socket is None:
                slave_socket = connect_to_slave()
                if not slave_socket: time.sleep(2); continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with lock: rgb_frame = latest_rgb_frame.copy() if latest_rgb_frame is not None else None
            tof_depth_frame = request_and_receive_tof_frames(slave_socket) 

            if rgb_frame is None or tof_depth_frame is None:
                logger.error("Không lấy được frame. Đóng kết nối để thử lại.");
                if slave_socket: slave_socket.close(); slave_socket = None
                continue

            all_keypoints, all_boxes = detector.run_detection(rgb_frame)
            logger.info(f"Detector đã chạy xong, phát hiện được {len(all_boxes)} người.")
            annotated_frame = rgb_frame.copy()

            for i, (box, kpts) in enumerate(zip(all_boxes, all_keypoints)):
                result = process_person(i, box, kpts, projector, tof_depth_frame, height_estimator, kalman_filters)
                
                # Vẽ kết quả lên ảnh annotated
                box_int = tuple(map(int, box))
                if result:
                    cv2.rectangle(annotated_frame, box_int[:2], box_int[2:], result['color'], 2)
                    cv2.putText(annotated_frame, result['text'], (box_int[0], box_int[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, result['color'], 2)
                
                # <<< TỐI ƯU: LƯU ẢNH CROP CỦA TỪNG NGƯỜI >>>
                xmin, ymin, xmax, ymax = box_int
                person_crop = rgb_frame[ymin:ymax, xmin:xmax]
                if person_crop.size > 0:
                    crop_filename = f"{timestamp}_P{i+1}.png"
                    cv2.imwrite(os.path.join(RGB_CROPS_DIR, crop_filename), person_crop)


            active_ids = set(range(len(all_boxes)))
            for stale_id in list(kalman_filters.keys() - active_ids): del kalman_filters[stale_id]

            cv2.imwrite(os.path.join(ANNOTATED_RGB_DIR, f"{timestamp}.png"), annotated_frame)
            logger.info(f"Đã xử lý và lưu ảnh kết quả: {timestamp}.png")
    finally:
        logger.info("Dọn dẹp tài nguyên..."); stop_event.set() 
        if slave_socket: slave_socket.close()
        if 'reader_thread' in locals() and reader_thread.is_alive(): reader_thread.join(timeout=1.0)
        cap_rgb.release(); detector.release(); logger.info("Ứng dụng đã đóng.")

if __name__ == '__main__':
    main()  