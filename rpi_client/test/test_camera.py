import cv2
from utils.config_camera import CameraHandler
import os  # Thêm import os 

def test_cam():
    # Tạo thư mục images nếu chưa tồn tại 
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Đã tạo thư mục images")

    # Tạo đối tượng CameraHandler 
    # cam = CameraHandler(camera_index=0) 
    cam = CameraHandler(camera_index=1) 

    try:
        # Lấy ảnh lores
        lores_frame = cam.capture_lores_frame() 
        if lores_frame is not None:
            print("Ảnh lores đã được chụp thành công.")
            # Lưu ảnh lores
            cv2.imwrite("images/lores_frame.jpg", lores_frame)
            print("Ảnh lores đã được lưu thành công.")
        else:
            print("Không thể chụp ảnh lores.")

        # Lấy ảnh main
        main_frame = cam.capture_main_frame()
        if main_frame is not None:
            print("Ảnh main đã được chụp thành công.") 
            # Lưu ảnh main 
            cv2.imwrite("images/main_frame.jpg", main_frame)
            print("Ảnh main đã được lưu thành công.") 
        else:
            print("Không thể chụp ảnh main.")

        # Lấy ảnh raw
        raw_frame = cam.capture_raw_frame()
        if raw_frame is not None:
            print("Ảnh raw đã được chụp thành công.")
            # Lưu ảnh raw 
            cv2.imwrite("images/raw_frame.tiff", raw_frame)
            print("Ảnh raw đã được lưu thành công.")
        else:
            print("Không thể chụp ảnh raw.")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi khi kiểm tra camera: {e}")
    
    finally:
        # Dừng camera sau khi kiểm tra
        cam.stop_camera()

# Chạy test
if __name__ == "__main__":
    test_cam()
