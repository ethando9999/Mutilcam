import cv2
import asyncio
import os
from camera import Camera  # Đảm bảo file Camera.py chứa class Camera như trên

async def run_camera_stream(cam, folder):
    os.makedirs(folder, exist_ok=True)
    # Chạy hàm start_stream trong thread riêng (không chặn event loop)
    await asyncio.to_thread(cam.start_stream, folder)

async def main():
    cam1 = Camera(device="/dev/video1", resolution="1K")
    cam2 = Camera(device="/dev/video3", resolution="1K") 
    
    # Tạo thư mục lưu ảnh
    folder_name = "calibration3"
    os.makedirs(folder_name, exist_ok=True)
    
    # Tạo task để chạy song song việc thu frame
    task1 = asyncio.create_task(run_camera_stream(cam1, f"{folder_name}/cam1"))
    task2 = asyncio.create_task(run_camera_stream(cam2, f"{folder_name}/cam2"))
    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    asyncio.run(main())
