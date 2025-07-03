import cv2
import numpy as np
import os
from utils.config_camera import CameraHandler  # Giả sử CameraHandler nằm trong camera_handler.py

# Tạo thư mục data nếu chưa tồn tại
os.makedirs('data', exist_ok=True)

# Khởi tạo CameraHandler
camera_handler = CameraHandler(0)

# Kiểm tra nếu camera đã được cấu hình mặc định
if not camera_handler.is_default_config:
    print("Không thể cấu hình camera.")
    exit()

# Định nghĩa thông số video 4K
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
fps = 30  # Số khung hình trên giây (có thể điều chỉnh)
resolution = (camera_handler.FRAME_WIDTH, camera_handler.FRAME_HEIGHT)  # Độ phân giải từ CameraHandler
output_file = 'data/output_4k_video.mp4'

# Khởi tạo VideoWriter
out = cv2.VideoWriter(output_file, fourcc, fps, resolution)

print("Chương trình đang chạy... Nhấn 'q' hoặc Ctrl+C để dừng.")

frame_count = 0

try:
    while True: 
        frame = camera_handler.capture_main_frame()
        
        if frame is None:
            print("Không thể đọc frame từ camera.")
            break
        
        # Ghi frame vào video
        out.write(frame)
        
        frame_count += 1
        print(f"Đã ghi frame thứ {frame_count}")
        
        # Thoát nếu nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nĐã nhận Ctrl+C. Đang lưu video và thoát...")
    out.release()  # Lưu video trước khi thoát
    cv2.destroyAllWindows()
    camera_handler.stop_camera()
    print(f"Đã ghi tổng cộng {frame_count} frame vào video {output_file}. Chương trình kết thúc.")
    exit()

# Giải phóng tài nguyên nếu thoát bình thường
out.release()
cv2.destroyAllWindows()
camera_handler.stop_camera()

print(f"Đã ghi tổng cộng {frame_count} frame vào video {output_file}. Chương trình kết thúc.")