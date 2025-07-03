from utils.config_camera import CameraHandler
import cv2
import time
from datetime import datetime

def capture_and_save_image():
    try:
        # Khởi tạo camera
        camera = CameraHandler()
        
        # Đợi camera khởi động 
        print("Đang khởi động camera...")
        time.sleep(2)
        
        # Biến đếm frame và thời gian
        frame_count = 0
        fps_count = 0
        start_time = time.time()
        last_time = start_time
        duration = 10  # Thời gian chạy (giây)
        
        print("Bắt đầu đo FPS...")
        while time.time() - start_time < duration:
            # Chụp ảnh
            frame = camera.capture_main_frame()
            
            # Tăng số frame
            frame_count += 1
            fps_count += 1
            
            # Tính FPS sau mỗi giây
            current_time = time.time()
            if current_time - last_time >= 1.0:  # Đã đủ 1 giây
                fps = fps_count / (current_time - last_time)
                print(f"Giây thứ {int(current_time - start_time)}: FPS = {fps:.2f}")
                fps_count = 0  # Reset fps_count sau mỗi giây
                last_time = current_time
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        # Đóng camera
        camera.stop_camera()
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