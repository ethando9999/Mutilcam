# master_trigger.py (version 5.1 - Correct Resolution for Calibration)
import socket
import os
import cv2
import time
from datetime import datetime
import threading

# --- CẤU HÌNH CHUNG ---
SLAVE_IP = "192.168.100.2"
TCP_PORT = 5005
CAPTURE_COUNT = 3000
CAPTURE_DELAY = 2

# --- CẤU HÌNH CAMERA RGB ---
# GIAI ĐOẠN 1: THU THẬP DỮ LIỆU CALIBRATION
# Ở giai đoạn này, chúng ta chọn một độ phân giải ĐƯỢC CAMERA HỖ TRỢ TỐT NHẤT
# và có tỉ lệ khung hình gần vuông nhất có thể. 640x480 (tỉ lệ 4:3) là lựa chọn
# tiêu chuẩn và tối ưu cho việc này.
# **LƯU Ý:** Đây là độ phân giải CHỤP, không phải kích thước ĐẦU VÀO của model YOLO.
RGB_DEVICE_ID = "/dev/video0" 
RGB_RESOLUTION = (640, 480) # (width, height)

# --- THƯ MỤC LƯU TRỮ ---
RGB_DIR = "rgb_frames"

# --- BIẾN TOÀN CỤC CHO THREADING ---
latest_frame = None
lock = threading.Lock() 
stop_event = threading.Event() 
 
def frame_reader(cap):
    """Luồng riêng biệt liên tục đọc camera để buffer luôn mới."""
    global latest_frame, lock, stop_event
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01)
    print("Luồng đọc camera đã dừng.")

def connect_to_slave(max_retries=5, retry_delay=3):
    """Hàm chuyên dụng để kết nối đến Slave, với cơ chế thử lại."""
    for attempt in range(max_retries):
        try:
            print(f"Đang kết nối đến Slave tại {SLAVE_IP}:{TCP_PORT}... (Lần thử {attempt + 1}/{max_retries})")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10.0)
            s.connect((SLAVE_IP, TCP_PORT))
            print(">>> Kết nối thành công! <<<\n")
            return s
        except (socket.timeout, ConnectionRefusedError) as e:
            print(f"!! Kết nối thất bại: {e}. Thử lại sau {retry_delay} giây...")
            if attempt + 1 == max_retries:
                return None
            time.sleep(retry_delay)

def main():
    global latest_frame, lock, stop_event
    
    print(f"Khởi động Master Trigger v{5.1}...")
    os.makedirs(RGB_DIR, exist_ok=True)
    print(f"Sẽ lưu ảnh RGB vào: '{RGB_DIR}'")

    cap = cv2.VideoCapture(RGB_DEVICE_ID)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera RGB tại ID {RGB_DEVICE_ID}.")
        return

    # Tối ưu: Tự động set độ phân giải từ cấu hình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_RESOLUTION[1])
    # Đọc lại để xác nhận
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera RGB đã sẵn sàng. Yêu cầu: {RGB_RESOLUTION[0]}x{RGB_RESOLUTION[1]}. Độ phân giải thực tế: {int(w)}x{int(h)}")

    reader_thread = threading.Thread(target=frame_reader, args=(cap,))
    reader_thread.daemon = True
    reader_thread.start()
    print("Luồng đọc camera đã được khởi động.")
    time.sleep(2)

    s = connect_to_slave()
    if not s:
        print("Lỗi: Không thể thiết lập kết nối với Slave. Đang thoát.")
        stop_event.set()
        reader_thread.join()
        cap.release()
        return 

    try:
        with s:
            for i in range(1, CAPTURE_COUNT + 1): 
                print(f"--- Chuẩn bị bộ ảnh {i}/{CAPTURE_COUNT} ---")
                input("-> Đặt bàn cờ vào vị trí và nhấn ENTER để chụp...")

                # 1. Gửi lệnh CAPTURE trước
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Gửi lệnh CAPTURE đến Slave...")
                s.sendall(b"CAPTURE")

                # 2. NGAY LẬP TỨC lưu frame RGB hiện tại
                # Đây là thay đổi quan trọng nhất để tối ưu đồng bộ
                with lock:
                    current_frame = latest_frame.copy() if latest_frame is not None else None
                
                if current_frame is None:
                    print("!! Lỗi: Chưa có frame nào từ luồng đọc camera. Bỏ qua...")
                    # Nếu không có frame, chúng ta vẫn cần đợi phản hồi từ slave để không làm loạn chu trình
                    s.recv(1024) 
                    time.sleep(1)
                    continue

                rgb_filename = os.path.join(RGB_DIR, f"rgb_{i:03d}.png")
                cv2.imwrite(rgb_filename, current_frame)
                print(f">>> Đã lưu (phía Master): {rgb_filename}")

                # 3. Bây giờ mới chờ phản hồi từ Slave để xác nhận
                print("-> Đang chờ Slave hoàn tất...")
                try:
                    # Đặt timeout cho recv để tránh bị treo vô hạn
                    s.settimeout(15.0) 
                    response = s.recv(1024).decode('utf-8')
                    s.settimeout(None) # Tắt timeout sau khi nhận xong
                except socket.timeout:
                    print("!! Lỗi: Slave không phản hồi trong 15 giây. Có thể Slave đã gặp sự cố.")
                    break # Thoát vòng lặp

                if not response:
                    print("!! Lỗi: Mất kết nối với Slave.") 
                    break
                
                if response == "DONE":
                    print(f"-> Slave xác nhận ĐÃ LƯU bộ ảnh {i}.")
                else:
                    print(f"!! Slave báo lỗi. Bộ ảnh {i} có thể không đồng bộ.")

                if i < CAPTURE_COUNT:
                    print(f"-> Chờ {CAPTURE_DELAY} giây trước khi tiếp tục...\n")
                    time.sleep(CAPTURE_DELAY)

    except (socket.timeout, ConnectionAbortedError):
        print("!! Lỗi: Slave không phản hồi hoặc mất kết nối.")
    except KeyboardInterrupt:
        print("\nPhát hiện Ctrl+C. Đang thoát...")
    finally:
        print("\nBắt đầu quá trình dọn dẹp...")
        stop_event.set()
        if s: s.close()
        if reader_thread.is_alive(): reader_thread.join()
        cap.release()
        print("Hoàn tất. Chương trình đã đóng.")

if __name__ == "__main__":
    main()