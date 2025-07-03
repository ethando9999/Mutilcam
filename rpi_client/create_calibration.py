import cv2
import asyncio
import os
from utils.config_camera import CameraHandler  # Đảm bảo file Camera.py chứa class Camera như trên
from utils.camera_log import setup_logging, get_logger

# Thiết lập logger khi khởi động
setup_logging()
logger = get_logger(__name__)

def capture_and_save(cam: CameraHandler, filepath):  
    frame = cam.capture_main_frame()
    cv2.imwrite(filepath, frame)
    logger.info(f"Đã lưu frame {filepath} thành công!")

async def run_camera_stream(cam, filepath):
    # Chạy hàm start_stream trong thread riêng (không chặn event loop)
    await asyncio.to_thread(capture_and_save, cam, filepath)

async def main():
    cam1 = CameraHandler(camera_index=1)
    cam2 = CameraHandler(camera_index=0)
    
    # Tạo thư mục lưu ảnh
    folder_name = "calibration8" 
    cam1_folder = f"{folder_name}/cam1"
    cam2_folder = f"{folder_name}/cam2"
    os.makedirs(folder_name, exist_ok=True) 
    os.makedirs(cam1_folder, exist_ok=True) 
    os.makedirs(cam2_folder, exist_ok=True) 
    frame_number = 0
    while True:
        frame_number +=1
        filename = f"frame_{frame_number}.jpg"
        # Tạo task để chạy song song việc thu frame
        task1 = asyncio.create_task(run_camera_stream(cam1, f"{cam1_folder}/{filename}"))
        task2 = asyncio.create_task(run_camera_stream(cam2, f"{cam2_folder}/{filename}"))
        await asyncio.gather(task1, task2)
# async def main():
#     cam1 = CameraHandler(camera_index=0)
#     await run_camera_stream(cam1, "cam1.jpg")
#     cam1.stop_camera()

#     await asyncio.sleep(1)

#     cam2 = CameraHandler(camera_index=1)
#     await run_camera_stream(cam2, "cam2.jpg")
#     cam2.stop_camera()


if __name__ == "__main__":
    asyncio.run(main())
