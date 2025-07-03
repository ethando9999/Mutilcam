# dual_camera_mjpeg.py
import cv2
import time
import signal
import sys
import logging
import os
import numpy as np

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def check_gstreamer_support():
    """
    Kiểm tra xem OpenCV có hỗ trợ GStreamer hay không.
    """
    build_info = cv2.getBuildInformation()
    if "GStreamer:                   YES" in build_info:
        logging.info("OpenCV được build với hỗ trợ GStreamer.")
        return True
    else:
        logging.error("OpenCV không được build với hỗ trợ GStreamer. Vui lòng build lại OpenCV với WITH_GSTREAMER=ON.")
        return False

def is_valid_frame(frame):
    """
    Kiểm tra xem frame có phải là frame đen (hoặc gần đen) hay không.
    """
    if frame is None or frame.size == 0:
        return False 
    mean_val = np.mean(frame)
    std_val = np.std(frame)
    # Giả sử frame hợp lệ nếu trung bình > 20 và độ lệch chuẩn > 10
    return mean_val > 20 and std_val > 10

# def open_camera(device, warmup_frames=10, use_mjpeg=True):
#     """
#     Mở camera theo device với pipeline cố định:
#       - Nếu use_mjpeg=True: dùng pipeline MJPEG (dành cho /dev/video1)
#       - Nếu use_mjpeg=False: dùng pipeline JPEG (dành cho /dev/video3)
#     Thực hiện bước warm-up để đảm bảo camera đã sẵn sàng.
#     """
#     if use_mjpeg:
#         # Pipeline dùng MJPEG cho /dev/video1
#         pipeline = (
#             f"v4l2src device={device} ! "
#             "image/jpeg,width=640,height=480,format=JPEG ! "
#             "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
#             "appsink drop=true sync=false max-buffers=30"
#         )
#         logging.info(f"Thử pipeline MJPEG cho {device}...")
#     else:
#         # Pipeline cho /dev/video3 (JPEG như đã dùng)
#         pipeline = (
#             f"v4l2src device={device} ! "
#             "image/jpeg,width=640,height=480,format=JPEG ! "
#             "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
#             "appsink drop=true sync=false max-buffers=30"
#         )
#         logging.info(f"Thử pipeline JPEG cho {device}...")

#     cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#     if not cap.isOpened():
#         logging.error(f"Không thể mở camera tại {device} với pipeline hiện tại.")
#         return None

#     # Warm-up: Đọc liên tiếp một số frame cho đến khi có 3 frame hợp lệ liên tiếp
#     logging.info(f"Khởi động camera tại {device} (warm-up {warmup_frames} frame)...")
#     valid_count = 0
#     for i in range(warmup_frames):
#         ret, frame = cap.read()
#         if not ret:
#             logging.warning(f"Không đọc được frame warm-up {i+1} từ {device}.")
#             continue
#         if is_valid_frame(frame):
#             valid_count += 1
#             logging.info(f"Frame warm-up {i+1} từ {device} hợp lệ (liên tiếp: {valid_count}).")
#             if valid_count >= 3:
#                 logging.info(f"Camera {device} sẵn sàng sau {i+1} frame.")
#                 break
#         else:
#             valid_count = 0
#             logging.info(f"Frame warm-up {i+1} từ {device} không hợp lệ.")
#         time.sleep(0.1)
    
#     if valid_count < 3:
#         logging.error(f"Camera tại {device} không sẵn sàng sau warm-up.")
#         cap.release()
#         return None

#     # Reset con trỏ stream nếu cần
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     return cap

def open_camera(device, warmup_frames=10):
    logging.info(f"Thử mở camera {device} với V4L2...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.error(f"Không thể mở camera tại {device}.")
        return None
    
    # Đặt thông số (nếu cần)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warm-up
    logging.info(f"Khởi động camera {device} (warm-up {warmup_frames} frame)...")
    valid_count = 0
    for i in range(warmup_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.warning(f"Không đọc được frame warm-up {i+1} từ {device}.")
            continue
        if is_valid_frame(frame):
            valid_count += 1
            logging.info(f"Frame warm-up {i+1} hợp lệ (liên tiếp: {valid_count}).")
            if valid_count >= 3:
                logging.info(f"Camera {device} sẵn sàng sau {i+1} frame.")
                break
        else:
            valid_count = 0
        time.sleep(0.1)
    
    if valid_count < 3:
        logging.error(f"Camera {device} không sẵn sàng sau warm-up.")
        cap.release()
        return None
    return cap


def run_dual_cameras():
    # Kiểm tra hỗ trợ GStreamer
    # if not check_gstreamer_support():
    #     sys.exit(1)
        
    # Chỉ sử dụng /dev/video1 và /dev/video3
    device1 = "/dev/video0"  # Dùng pipeline MJPEG
    device2 = "/dev/video2"  # Dùng pipeline JPEG
    
    # Mở camera
    cap1 = open_camera(device1, warmup_frames=30)
    cap2 = open_camera(device2, warmup_frames=10)
    
    if cap1 is None or cap2 is None:
        logging.error("Không thể mở cả hai camera cùng lúc.")
        sys.exit(1)
    
    frame_count = 0
    start_time = time.time()
    running = True
    first_frame_saved = False

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        logging.info("Dừng chương trình...")
    signal.signal(signal.SIGINT, signal_handler)
    
    while running:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            logging.warning("Không đọc được frame từ một trong hai camera.")
            break
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        logging.info(f"Frame {frame_count}: FPS = {fps:.2f}")

        cv2.imshow(f"Camera {device1}", frame1)
        cv2.imshow(f"Camera {device2}", frame2)
        
        # if not first_frame_saved and is_valid_frame(frame1) and is_valid_frame(frame2):
        #     cv2.imwrite("video1_frame.jpg", frame1)
        #     cv2.imwrite("video3_frame.jpg", frame2)
        #     logging.info("Đã lưu frame đầu tiên hợp lệ từ cả 2 camera.")
        #     first_frame_saved = True
        
        time.sleep(0.01)
    
    cap1.release()
    cap2.release()
    logging.info(f"Đã giải phóng tài nguyên. FPS trung bình: {fps:.2f}")

if __name__ == "__main__":
    run_dual_cameras()
