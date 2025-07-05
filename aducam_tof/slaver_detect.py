# file: slaver_detect.py (Phiên bản đã tinh chỉnh)
import socket
import numpy as np
import time
import json
import struct
import logging

# Cấu hình logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import ArducamDepthCamera as ac
except ImportError:
    logger.warning("Không tìm thấy thư viện ArducamDepthCamera. Sẽ chạy ở chế độ giả lập.")
    ac = None

# --- CẤU HÌNH ---
HOST_IP = '0.0.0.0'
TCP_PORT = 5005
MAX_DISTANCE = 4000
CAMERA_TIMEOUT_MS = 200 # Đặt timeout của camera ra đây để dễ cấu hình

# Cấu hình cho chế độ giả lập
MOCK_FRAME_SHAPE = (180, 240)
MOCK_FRAME_DTYPE = np.uint16

def setup_camera():
    """Khởi tạo và cấu hình camera Arducam ToF."""
    if not ac:
        logger.error("Không thể khởi tạo camera vì thiếu thư viện ArducamDepthCamera.")
        return None
    try:
        cam = ac.ArducamCamera()
        if cam.open(ac.Connection.CSI, 0) != 0:
            raise RuntimeError("Không thể mở camera CSI.")
        # Yêu cầu cả hai frame depth và amplitude
        if cam.start(ac.FrameType.DEPTH_AMPLITUDE) != 0:
            raise RuntimeError("Không thể bắt đầu stream depth và amplitude.")
        cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
        logger.info(f"Camera ToF đã sẵn sàng. Cấu hình range: {MAX_DISTANCE}mm, timeout: {CAMERA_TIMEOUT_MS}ms.")
        return cam
    except Exception as e:
        logger.error(f"Lỗi khởi tạo camera: {e}")
        return None

def handle_client(conn: socket.socket, addr: tuple, cam):
    """Vòng lặp xử lý yêu cầu từ một client đã kết nối."""
    logger.info(f"Master đã kết nối từ: {addr}")
    is_mock_mode = (cam is None)

    while True:
        try:
            # Chờ lệnh từ Master. Bộ đệm 32 bytes là đủ cho lệnh "CAPTURE".
            data = conn.recv(32)
            if not data:
                logger.warning("Master đã đóng kết nối một cách bình thường.")
                break
            
            if data.strip() != b"CAPTURE":
                logger.warning(f"Nhận được dữ liệu không hợp lệ: {data}. Bỏ qua.")
                continue

            logger.info("Nhận được lệnh 'CAPTURE'. Đang xử lý...")
            
            depth_frame = None
            amp_frame = None

            if not is_mock_mode:
                frame = cam.requestFrame(CAMERA_TIMEOUT_MS)
                if frame:
                    # Sử dụng .copy() là một lựa chọn an toàn để đảm bảo dữ liệu không bị
                    # thay đổi bởi các tiến trình khác sau khi releaseFrame.
                    depth_frame = frame.depth_data.copy()
                    amp_frame = frame.amplitude_data.copy()
                    cam.releaseFrame(frame)
            else:
                logger.debug("Chế độ giả lập: Tạo frame giả.")
                depth_frame = np.random.randint(0, MAX_DISTANCE, size=MOCK_FRAME_SHAPE, dtype=MOCK_FRAME_DTYPE)
                amp_frame = np.random.randint(0, 1000, size=MOCK_FRAME_SHAPE, dtype=MOCK_FRAME_DTYPE)

            if depth_frame is not None and amp_frame is not None:
                header = {
                    'depth_shape': depth_frame.shape, 'depth_dtype': str(depth_frame.dtype),
                    'amp_shape': amp_frame.shape, 'amp_dtype': str(amp_frame.dtype)
                }
                header_bytes = json.dumps(header).encode('utf-8')
                
                # Gửi dữ liệu theo đúng giao thức Master yêu cầu
                conn.sendall(struct.pack('>I', len(header_bytes)))
                conn.sendall(header_bytes)
                conn.sendall(depth_frame.tobytes())
                conn.sendall(amp_frame.tobytes())
                
                logger.info(f"Đã gửi thành công frame (Depth: {depth_frame.shape}, Amp: {amp_frame.shape})")
            else:
                logger.error("Không nhận được frame từ camera. Gửi tín hiệu lỗi tới Master.")
                conn.sendall(struct.pack('>I', 0))

        except (ConnectionResetError, BrokenPipeError) as e:
            logger.warning(f"Mất kết nối với Master {addr}: {e}")
            break
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi xử lý client {addr}: {e}", exc_info=True) # exc_info=True để log cả traceback
            break
            
    logger.info(f"Đã đóng kết nối với {addr}")
    conn.close()

def main():
    logger.info("Khởi động Slave Listener...")
    cam = setup_camera()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST_IP, TCP_PORT))
    s.listen(1)
    logger.info(f"Đang chờ kết nối từ Master trên port {TCP_PORT}...")

    try:
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr, cam)
    except KeyboardInterrupt:
        logger.info("\nĐang đóng server...")
    finally:
        if cam:
            cam.close()
        s.close()
        logger.info("Server đã đóng.")

if __name__ == "__main__":
    main()