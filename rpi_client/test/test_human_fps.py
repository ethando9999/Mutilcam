from utils.config_camera import CameraHandler
from utils.yolo_pose import HumanDetection
# from utils.camera_log import setup_logging, get_logger
import cv2
import time
from datetime import datetime

# setup_logging()
# logger = get_logger(__name__)

def resize_frame(frame, target_width=640):
    """
    Resize frame với width cố định và giữ tỷ lệ khung hình.
    
    Args:
        frame: Khung hình đầu vào
        target_width: Chiều rộng mục tiêu (mặc định 640)
        
    Returns:
        Khung hình đã resize
    """
    height, width = frame.shape[:2]
    resize_ratio = target_width / width
    target_height = int(height * resize_ratio)
    return cv2.resize(frame, (target_width, target_height))

def capture_and_save_image():
    try:
        # Khởi tạo camera
        cam_13mp = CameraHandler(camera_index=0)
        # cam_5mp = CameraHandler(camera_index=1)
        human_detection = HumanDetection()
        
        # Đợi camera khởi động 
        print("Đang khởi động camera...")
        # time.sleep(2)

        # frame = cam_13mp.capture_main_frame()         
        # Biến đếm frame và thời gian
        frame_count = 0
        fps_count = 0
        start_time = time.time()
        last_time = start_time
        duration = 20  # Thời gian chạy (giây)
        # frame = cv2.imread("rpi_client/human3.jpg")

        
        print("Bắt đầu đo FPS...")
        while time.time() - start_time < duration:
            # Chụp ảnh
            frame = cam_13mp.capture_main_frame() 
            # frame = cam_5mp.capture_main_frame() 
            # resized_frame = resize_frame(frame, 1000)
            

            keypoints_data, boxes_data = human_detection.run_detection(frame)
            print('keypoints_data: ', keypoints_data) 

            # Tăng số frame
            frame_count += 1
            fps_count += 1
            
            # Tính FPS sau mỗi giây
            current_time = time.time()
            if current_time - last_time >= 1.0:  # Đã đủ 1 giây
                fps = fps_count / (current_time - last_time)
                # print(f"Giây thứ {int(current_time - start_time)}: FPS phát hiện người ảnh {cam_13mp.FRAME_WIDTH}x{cam_13mp.FRAME_HEIGHT}= {fps:.2f}")
                # print(f"Giây thứ {int(current_time - start_time)}: FPS pi5 cam 5mp lấy ảnh {cam_5mp.FRAME_WIDTH}x{cam_5mp.FRAME_HEIGHT} = {fps:.2f}")
                print(f"Giây thứ {int(current_time - start_time)}: FPS pi5 cam_13mp lấy ảnh {cam_13mp.FRAME_WIDTH}x{cam_13mp.FRAME_HEIGHT} = {fps:.2f}")
                # print(f"Giây thứ {int(current_time - start_time)}: FPS pi5 lấy ảnh cho cả 2 cam = {fps:.2f}")
                fps_count = 0  # Reset fps_count sau mỗi giây
                last_time = current_time
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        # Đóng camera
        cam_13mp.stop_camera()
        cam_5mp.stop_camera()
        print("Đã đóng camera")
        
        # In thông tin FPS cuối cùng
        if frame_count > 0:
            total_time = time.time() - start_time
            average_fps = frame_count / total_time
            print(f"\nKết quả đo trong {total_time:.1f} giây:")
            print(f"Tổng số frame: {frame_count}")
            print(f"FPS trung bình: {average_fps:.2f}")

if __name__ == "__main__":
    capture_and_save_image()