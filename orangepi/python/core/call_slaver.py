import numpy as np
import time
import socket
import json
import struct
import threading # Thêm import
from typing import Tuple, Optional

# ... (logger và hằng số giữ nguyên) ...
SLAVE_IP = "192.168.100.2"
TCP_PORT = 5005
# Giả lập logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CallSlave:
    # ... (__init__, connect, close, _receive_all không thay đổi) ...
    def __init__(self, slave_ip: str = SLAVE_IP, tcp_port: int = TCP_PORT, max_retries: int = 5, retry_delay: int = 3, timeout: float = 10.0, connect_on_init: bool = True):
        self.slave_ip = slave_ip
        self.tcp_port = tcp_port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        self.sock: Optional[socket.socket] = None
        self.logger = logger
        
        # --- THAY ĐỔI: Thêm Lock để đảm bảo an toàn luồng ---
        self.lock = threading.Lock()

        if connect_on_init:
            self.logger.info("Cố gắng kết nối ngay khi khởi tạo (connect_on_init=True)...")
            self.connect()

    def connect(self) -> bool:
        if self.sock:
            self.logger.debug("Kết nối đã tồn tại.")
            return True
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Đang kết nối đến Slave tại {self.slave_ip}:{self.tcp_port} (Lần thử {attempt + 1})...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                s.connect((self.slave_ip, self.tcp_port))
                self.sock = s
                self.logger.info(">>> Kết nối đến Slave thành công! <<<")
                return True
            except (socket.timeout, socket.error) as e:
                self.logger.warning(f"Kết nối thất bại: {e}. Thử lại sau {self.retry_delay} giây...")
                time.sleep(self.retry_delay)
        self.logger.error(f"Không thể kết nối đến Slave sau {self.max_retries} lần thử.")
        self.sock = None
        return False

    def close(self):
        if self.sock:
            self.logger.info("Đang đóng kết nối đến Slave.")
            try:
                self.sock.close()
            except socket.error as e:
                self.logger.error(f"Lỗi khi đóng socket: {e}")
            finally:
                self.sock = None

    def _receive_all(self, n: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < n:
            try:
                if self.sock is None:
                    self.logger.error("Socket đã bị đóng trong khi đang chờ nhận dữ liệu.")
                    return None
                packet = self.sock.recv(n - len(data))
                if not packet:
                    self.logger.warning("Kết nối bị đóng bởi phía Slave trong khi đang nhận dữ liệu.")
                    return None
                data.extend(packet)
            except socket.timeout:
                self.logger.error("Hết thời gian chờ (timeout) khi đang nhận dữ liệu.")
                return None
        return bytes(data)


    def request_and_receive_tof_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Gửi yêu cầu và nhận frame. Phương thức này giờ đã an toàn khi gọi từ nhiều luồng.
        """
        # --- THAY ĐỔI: Sử dụng lock để bảo vệ toàn bộ thao tác ---
        with self.lock:
            # Logic tự phục hồi: đảm bảo có kết nối trước khi thực hiện.
            if not self.sock:
                if not self.connect():
                    return None, None # Không thể kết nối, bỏ qua lần yêu cầu này.

            try:
                # Toàn bộ chu trình request-response được bảo vệ trong lock
                self.sock.sendall(b"CAPTURE")
                header_len_bytes = self._receive_all(4)
                if not header_len_bytes:
                    raise ConnectionError("Không nhận được độ dài header, kết nối có thể đã bị mất.")
                
                header_len = struct.unpack('>I', header_len_bytes)[0]
                if header_len == 0:
                    self.logger.error("Nhận được tín hiệu lỗi (header rỗng) từ Slave.")
                    return None, None
                
                header_bytes = self._receive_all(header_len)
                if not header_bytes:
                    raise ConnectionError("Không nhận được dữ liệu header.")
                
                header = json.loads(header_bytes.decode('utf-8'))

                depth_dtype = np.dtype(header['depth_dtype'])
                depth_shape = tuple(header['depth_shape'])
                depth_size = np.prod(depth_shape) * depth_dtype.itemsize
                depth_bytes = self._receive_all(depth_size)
                if not depth_bytes:
                    raise ConnectionError("Không nhận được dữ liệu depth frame.")
                depth_data = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape)

                amp_dtype = np.dtype(header['amp_dtype'])
                amp_shape = tuple(header['amp_shape'])
                amp_size = np.prod(amp_shape) * amp_dtype.itemsize
                amp_bytes = self._receive_all(amp_size)
                if not amp_bytes:
                    raise ConnectionError("Không nhận được dữ liệu amplitude frame.")
                amp_frame = np.frombuffer(amp_bytes, dtype=amp_dtype).reshape(amp_shape)

                return depth_data, amp_frame

            except (ConnectionError, socket.error, BrokenPipeError) as e:
                self.logger.error(f"Lỗi trong quá trình giao tiếp: {e}. Đóng kết nối.")
                self.close()
                return None, None
            except Exception as e:
                self.logger.error(f"Một lỗi không mong muốn xảy ra: {e}")
                self.close()
                return None, None

    def __enter__(self):
        # Không cần thay đổi
        if not self.sock:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Không cần thay đổi
        self.close()


# --- CÁCH SỬ DỤNG (ĐÃ CẬP NHẬT VỚI THỐNG KÊ) ---
if __name__ == "__main__":
    
    print("Khởi tạo slave (có thể bị chậm một chút để kết nối)...")
    slave = CallSlave(SLAVE_IP, TCP_PORT, max_retries=2, retry_delay=1)
    
    if slave.sock:
        print(">>> Kết nối thành công ngay khi khởi tạo! <<<\n")
    else:
        print(">>> Kết nối khi khởi tạo thất bại. Sẽ thử lại khi gọi `request`. <<<\n")

    # THÊM: Danh sách để lưu trữ thời gian của các lần thành công
    successful_timings = []
    num_requests = 10 # Tăng số lần lặp để thống kê chính xác hơn

    # Vòng lặp yêu cầu frame
    for i in range(num_requests):
        print(f"Yêu cầu frame lần {i + 1}/{num_requests}...")
        
        start_time = time.perf_counter()
        depth_data, amp_frame = slave.request_and_receive_tof_frames()
        end_time = time.perf_counter()
        duration = end_time - start_time

        if depth_data is not None and amp_frame is not None:
            print(f"  > Đã nhận thành công! Thời gian: {duration:.4f} giây")
            # THÊM: Lưu lại thời gian nếu thành công
            successful_timings.append(duration)
        else:
            print(f"  > Không nhận được frame. Thời gian chờ/lỗi: {duration:.4f} giây")
        
        time.sleep(0.5)

    print("\n--- THỐNG KÊ HIỆU NĂNG ---")
    if successful_timings:
        num_success = len(successful_timings)
        avg_time = sum(successful_timings) / num_success
        min_time = min(successful_timings)
        max_time = max(successful_timings)
        
        print(f"Số lần yêu cầu thành công: {num_success}/{num_requests}")
        print(f"Thời gian trung bình: {avg_time:.4f} giây")
        print(f"Thời gian nhanh nhất (Min): {min_time:.4f} giây")
        print(f"Thời gian chậm nhất (Max): {max_time:.4f} giây")
    else:
        print("Không có yêu cầu nào thành công để thống kê.")

    print("\nĐóng kết nối.")
    slave.close()
