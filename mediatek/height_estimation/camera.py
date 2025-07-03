import cv2
import time
import threading
import os

RESOLUTIONS = {
    "1K": (1280, 720),    # 720p
    "2K": (2560, 1440),   # 1440p
    "3K": (3200, 1800),   # Gần 3K
    "4K": (3840, 2160),   # 2160p
    "5MP": (2592, 1944),  # 5 Megapixels
    "640": (640, 480), 
}

class Camera:
    def __init__(self, device, resolution):
        self.device = device
        self.resolution = resolution
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.running = True
        self.set_resolution(resolution)
        # self.key_listener_thread = threading.Thread(target=self.key_listener, daemon=True)
        # self.key_listener_thread.start()


    
    def set_resolution(self, resolution_name):
        if resolution_name in RESOLUTIONS:
            width, height = RESOLUTIONS[resolution_name]
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width == width and actual_height == height:
                print(f"Đã đặt độ phân giải: {resolution_name} ({width}x{height})")
            else:
                print(f"Không thể đặt {resolution_name}. Độ phân giải thực tế: {actual_width}x{actual_height}")
        else:
            print("Độ phân giải không hợp lệ.")

    def start_stream(self, folder):
        frame_number = 0 
        prev_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Không thể đọc frame từ camera {self.device}")
                break
            self.frame = frame
            frame_number +=1

            # cv2.imshow(f"Camera {self.device}", frame)
            # Tính FPS (không dùng hiển thị log ở đây để giảm tải)
            current_time = time.time()
            # if current_time - prev_time > 0:
            #     fps = 1 / (current_time - prev_time)
            #     print(f"FPS device{self.device}: {fps}")
                # prev_time = current_time
            if frame_number > 20 and (current_time - prev_time) > 1:
                filename = f"frame_{frame_number}.jpg"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, self.frame)
                print(f"lưu ảnh vào {filepath} thành công!")
                prev_time = current_time
            # Có thể lưu log FPS nếu cần
        self.cap.release()


    def key_listener(self, filename="captured_image.jpg"):
        from pynput import keyboard
        import time

        print("Nhấn phím space để chụp hình.", flush=True)

        def on_press(key):
            # Kiểm tra nếu phím space được nhấn
            if key == keyboard.Key.space:
                print("Bắt đầu chụp!", flush=True)
                cv2.imwrite(filename, self.frame)

        # Khởi tạo listener cho bàn phím
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def stop(self):
        self.running = False
