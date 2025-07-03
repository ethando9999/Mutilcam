import logging
import socket
import ssl
import cv2
import time
import zlib
import threading
from pi_client.utils.rmbg_mog import BackgroundRemover
from utils.yolo_pose import HumanDetection

# Cấu hình logging
LOG_FILE = "cam.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Ghi log vào file
        logging.StreamHandler()        # Hiển thị log trong terminal
    ]
)

# Cấu hình kết nối
SERVER_IP = "center.pi"
SERVER_PORT = 9090
CHUNK_SIZE = 1024
END_CHUNK = b"END"
CERT_FILE = "server.crt"
MAX_RETRY = 3

lock = threading.Lock()

def calculate_checksum(data):
    """Tính checksum 4 byte của dữ liệu."""
    return zlib.crc32(data)

def connect_to_server():
    """Kết nối tới server với xác minh SSL."""
    retry_count = 0
    while retry_count < MAX_RETRY:
        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.load_verify_locations(CERT_FILE)

            client_socket = socket.create_connection((SERVER_IP, SERVER_PORT))
            ssl_socket = context.wrap_socket(client_socket, server_hostname=SERVER_IP)

            logging.info("Connected to server with SSL.")
            return ssl_socket
        except (socket.error, ssl.SSLError) as e:
            retry_count += 1
            logging.error(f"Connection failed: {e}. Retrying ({retry_count}/{MAX_RETRY})...")
    return None

def send_frame(client_socket, frame):
    try:
        frame_size = len(frame)
        sequence_number = 1

        for i in range(0, frame_size, CHUNK_SIZE):
            chunk = frame[i:i + CHUNK_SIZE]
            checksum = calculate_checksum(chunk)
            sequence = sequence_number.to_bytes(4, 'big')
            checksum_bytes = checksum.to_bytes(4, 'big')
            message = sequence + checksum_bytes + chunk

            client_socket.sendall(message)
            sequence_number += 1

        # Gửi chuỗi kết thúc
        sequence_number = 0
        checksum = 0
        sequence = sequence_number.to_bytes(4, 'big')
        checksum_bytes = checksum.to_bytes(4, 'big')
        message = sequence + checksum_bytes
        client_socket.sendall(message)
        logging.info("Frame sent successfully.")
        return True
    except (socket.error, ssl.SSLError, ssl.SSLEOFError) as e:
        logging.error(f"Transmission error: {e}")
        return False

def retry_send(client_socket, frame):
    """Thử gửi lại frame nếu lỗi."""
    for attempt in range(MAX_RETRY):
        if send_frame(client_socket, frame):
            return True
        logging.warning(f"Retrying frame transmission {attempt + 1}/{MAX_RETRY}...")
    return False

def main():
    """Quay video và gửi frame."""
    client_socket = connect_to_server()
    if not client_socket:
        logging.error("Failed to connect to server. Exiting.")
        return

    background_remover = BackgroundRemover()
    human_detection = HumanDetection()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot access camera. Exiting.")
        return

    # Biến lưu thời gian bắt đầu và số frame
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from camera. Retrying...")
            time.sleep(1)
            continue

        try:
            # Human Detection
            keypoints_data, boxes_data = human_detection.run_detection(frame)
            logging.info(f"Detected {len(boxes_data)} humans.")

            for human_id, box in enumerate(boxes_data):
                x1, y1, x2, y2 = map(int, box)
                human_box = frame[y1:y2, x1:x2].copy()
                height, width, _ = human_box.shape
                logging.info(f"Human box size: {width}x{height}")

                # Nén frame thành JPEG
                _, encoded_box = cv2.imencode('.jpg', human_box)
                box_data = encoded_box.tobytes()

                # Gửi frame và xử lý lỗi
                if not retry_send(client_socket, box_data):
                    logging.warning("Reconnecting to server...")
                    if client_socket is not None:
                        client_socket.close()
                        client_socket = None
                    client_socket = connect_to_server()
                    if not client_socket:
                        logging.error("Failed to reconnect. Exiting.")
                        break

            # Cập nhật frame_count và tính FPS trung bình mỗi giây
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:  # Tính FPS trung bình mỗi giây
                fps = frame_count / elapsed_time
                logging.info(f"FPS: {fps:.2f}")
                start_time = current_time
                frame_count = 0

        except Exception as e:
            logging.exception(f"Unexpected error: {e}")

    logging.info("Video capture and transmission stopped.")
    cap.release()
    if client_socket:
        client_socket.close()

if __name__ == "__main__":
    main()
