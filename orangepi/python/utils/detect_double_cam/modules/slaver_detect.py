# file: slaver_detect.py (v12 - Giao thức cuối cùng)
import socket
import numpy as np
import time
import json
import struct
import threading
import os

from logging_python import setup_logging, get_logger
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
setup_logging(log_dir=LOG_DIR, log_file="slaver_detect.log")
logger = get_logger(__name__)

try:
    import ArducamDepthCamera as ac
    logger.info("Thư viện ArducamDepthCamera đã được import thành công.")
except ImportError:
    logger.warning("Không tìm thấy thư viện ArducamDepthCamera. Chạy ở chế độ mô phỏng.")
    ac = None

HOST_IP = '0.0.0.0'
TCP_PORT = 5005

def setup_camera():
    if not ac: return None
    try:
        cam = ac.ArducamCamera()
        if cam.open(ac.Connection.CSI, 0) != 0: raise RuntimeError("Không thể mở camera CSI.")
        if cam.start(ac.FrameType.DEPTH) != 0: raise RuntimeError("Không thể bắt đầu stream depth.")
        cam.setControl(ac.Control.RANGE, 4000)
        logger.info("Camera ToF đã sẵn sàng.")
        return cam
    except Exception as e:
        logger.error(f"Khởi tạo camera ToF thất bại: {e}", exc_info=True); return None

def create_dummy_frame():
    logger.info("Tạo frame ToF giả để mô phỏng.")
    return np.random.randint(500, 4000, size=(180, 240), dtype=np.uint16)

def handle_client(conn, addr, cam):
    logger.info(f"Master đã kết nối từ: {addr}")
    try:
        while True:
            data = conn.recv(1024)
            if not data or data.decode('utf-8', errors='ignore') != "CAPTURE":
                logger.warning(f"Master {addr} đã đóng kết nối hoặc gửi lệnh không hợp lệ.")
                break
            
            logger.info(f"Nhận được lệnh CAPTURE từ {addr}")
            depth_data = create_dummy_frame() if not cam else None
            if cam:
                frame = cam.requestFrame(200)
                if frame:
                    depth_data = frame.depth_data.copy()
                    cam.releaseFrame(frame)
            
            if depth_data is not None:
                header = {'depth_shape': depth_data.shape, 'depth_dtype': str(depth_data.dtype)}
                header_bytes = json.dumps(header).encode('utf-8')
                depth_bytes = depth_data.tobytes()

                # <<< GIAO THỨC GỬI DỮ LIỆU ĐỒNG BỘ >>>
                conn.sendall(struct.pack('>I', len(header_bytes)))
                conn.sendall(header_bytes)
                conn.sendall(depth_bytes)
                logger.info(f"Đã gửi frame {depth_data.shape} ({len(depth_bytes)} bytes) tới {addr}")
            else:
                logger.error("Không nhận được frame ToF từ camera.")
                conn.sendall(struct.pack('>I', 0))

    except (ConnectionResetError, BrokenPipeError):
        logger.warning(f"Master {addr} đã ngắt kết nối đột ngột.")
    except Exception as e:
        logger.error(f"Lỗi khi xử lý client {addr}: {e}", exc_info=True)
    finally:
        logger.info(f"Đóng kết nối với {addr}")
        conn.close()

def main():
    logger.info("Khởi động Slave Listener (v12 - Giao thức cuối cùng)...")
    cam = setup_camera()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST_IP, TCP_PORT))
    s.listen(5)
    logger.info(f"Đang chờ kết nối trên port {TCP_PORT}...")
    try:
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, cam))
            client_thread.start()
    except KeyboardInterrupt:
        logger.info("\nĐang đóng server...")
    finally:
        if cam: cam.close()
        s.close()
        logger.info("Server đã đóng.")

if __name__ == "__main__":
    main()