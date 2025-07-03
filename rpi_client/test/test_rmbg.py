from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import socket
import zlib
import time
import queue
import random
# import logging
import os
import platform
import asyncio


import cv2
from utils.config_camera import CameraHandler
from utils.rmbg_mog import BackgroundRemover
from utils.camera_log import logging


camerahandler = CameraHandler()
background_remover = BackgroundRemover()

# Đầu file, thay đổi cấu hình logging 
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cấu hình kết nối
SERVER_IP = "192.168.7.1" 
SERVER_PORT = 5050 
MIN_CHUNK_SIZE = 4096
CHUNK_SIZE = MIN_CHUNK_SIZE
PASSWORD = "secret_password"
AES_KEY = sha256(PASSWORD.encode()).digest()
INIT_FLAG = 1
END_FLAG = 2
ERROR_FLAG = 3

MAX_RETRY = 3  # Giới hạn số lần retry

# Tạo IV dùng chung cho cả ứng dụng
iv = get_random_bytes(16)


def log_cpu_temp():
    """Ghi lại nhiệt độ CPU tùy theo hệ điều hành."""
    try:
        system = platform.system()  # Lấy thông tin hệ điều hành

        if system == "Linux":
            # Kiểm tra nếu là Raspberry Pi
            if os.path.exists("/usr/bin/vcgencmd"):
                temp = os.popen("vcgencmd measure_temp").readline().strip()
                logging.info(f"CPU Temperature (Raspberry Pi): {temp}")
            else:
                # Debian/Ubuntu
                temp = os.popen("cat /sys/class/thermal/thermal_zone0/temp").readline().strip()
                temp_c = int(temp) / 1000  # Chuyển đổi từ millidegree Celsius sang Celsius
                logging.info(f"CPU Temperature (Linux): {temp_c:.1f}°C")
        elif system == "Darwin":
            logging.info(f"CPU Thermal Level (macOS): n/a")
        else:
            logging.warning(f"Unsupported system: {system}. Unable to log CPU temperature.")
    except Exception as e:
        logging.error(f"Error reading CPU temperature: {e}")

def aes_encrypt(data):
    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv=iv)
    padded_data = pad(data, AES.block_size)
    
    return iv + cipher.encrypt(padded_data)

def aes_decrypt(data):
    iv, data = data[:16], data[16:]
    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv=iv)
    return unpad(cipher.decrypt(data), AES.block_size)

def calculate_checksum(frame_number, sequence, data, uuid):
    """
    Tính checksum của chunk hoặc payload, bao gồm UUID để đảm bảo tính duy nhất.
    """
    payload = (
        frame_number.to_bytes(1, 'big') +
        sequence.to_bytes(2, 'big') +
        data +
        uuid.encode() +
        PASSWORD.encode()
    )
    return zlib.crc32(payload) % 65536


def generate_uuid():
    """
    Tạo chuỗi UUID ngẫu nhiên 6 chữ số.
    """
    return f"{random.randint(100000, 999999):06}"


def send_init_sequence(udp_socket, frame_number, uuid, chunk_size, total_chunks, server_address):
    """
    Gửi gói tin "init" để bắt đầu gửi frame.
    """
    logging.info(f"Start init sequence with UUID {uuid} to server.")
    sequence_number = 0
    encrypted_payload = (
        INIT_FLAG.to_bytes(1, 'big') +
        aes_encrypt(
            uuid.encode() +
            chunk_size.to_bytes(2, 'big') +
            total_chunks.to_bytes(2, 'big')
        )
    )
    checksum = calculate_checksum(frame_number, sequence_number, encrypted_payload, uuid)
    header = frame_number.to_bytes(1, 'big') + sequence_number.to_bytes(2, 'big') + checksum.to_bytes(2, 'big')
    message = header + encrypted_payload
    udp_socket.sendto(message, server_address)
    # logging.info(f"Successfully sent init message to {server_address} for frame number: {frame_number}") 
    # Chờ phản hồi từ server
    udp_socket.settimeout(5.0)
    try:
        data, addr = udp_socket.recvfrom(1024)
        if data == b"OK":
            return True
    except socket.timeout:
        logging.error("Timeout waiting for server acknowledgment. Aborting frame.")
    return False


def send_end_sequence(udp_socket, frame_number, uuid, frame_size, combined_checksums, server_address, error=False):
    """
    Gửi sequence đặc biệt để báo kết thúc việc gửi frame.
    """
    sequence_number = 0

    encrypted_payload = b""
    if error:
        # payload chứa "error" để báo lỗi
        encrypted_payload = (
            ERROR_FLAG.to_bytes(1, 'big') +
            aes_encrypt(
                uuid.encode() +
                frame_size.to_bytes(4, 'big') +  # Frame size dùng 4 bytes
                b"error"
            )     
        )
    else:
        # Tổng checksum là giá trị combined_checksums đã tích lũy
        encrypted_payload = (
            END_FLAG.to_bytes(1, 'big') +
            aes_encrypt(
                uuid.encode() +
                frame_size.to_bytes(4, 'big') +  # Frame size dùng 4 bytes
                combined_checksums.to_bytes(2, 'big')
            )
        )

    # encrypted_payload = aes_encrypt(payload)
    checksum = calculate_checksum(frame_number, sequence_number, encrypted_payload, uuid)
    header = frame_number.to_bytes(1, 'big') + sequence_number.to_bytes(2, 'big') + checksum.to_bytes(2, 'big')
    message = header + encrypted_payload
    udp_socket.sendto(message, server_address)


def send_frame(frame_number, udp_socket, frame, server_address):
    """
    Gửi một frame qua UDP và nhận phản hồi trực tiếp từ server.
    """
    logging.info(f"Sending frame {frame_number} to server.")
    frame_size = len(frame)  # Kích thước thực tế của frame

    adjust_chunk_size(frame_size)  # Điều chỉnh CHUNK_SIZE nếu cần

    total_chunks = (frame_size + CHUNK_SIZE - 1) // CHUNK_SIZE  # Số lượng chunk
    uuid = generate_uuid()  # Tạo chuỗi UUID ngẫu nhiên

    
    combined_checksums = 0  # Tích lũy checksum của từng chunk

    # Gửi gói tin init
    if not send_init_sequence(udp_socket, frame_number, uuid, CHUNK_SIZE, total_chunks, server_address):
        return False  # Hủy gửi frame nếu không nhận được phản hồi từ server

    sent_chunks = {}

    for sequence in range(1, total_chunks + 1):
        start_idx = (sequence - 1) * CHUNK_SIZE
        chunk = frame[start_idx:start_idx + CHUNK_SIZE]
        
        # Giải mã cho chunk đầu tiên và chunk thứ 10
        if sequence == 1 or sequence == 10:
            encrypted_chunk = aes_encrypt(chunk[:128]) + chunk[128:]
        else:
            encrypted_chunk = chunk

        checksum = calculate_checksum(frame_number, sequence, encrypted_chunk, uuid)
        header = frame_number.to_bytes(1, 'big') + sequence.to_bytes(2, 'big') + checksum.to_bytes(2, 'big')
        message = header + encrypted_chunk
        udp_socket.sendto(message, server_address)

        combined_checksums = (combined_checksums + checksum) % 65536
        sent_chunks[sequence] = message
    # Gửi sequence đặc biệt để báo kết thúc frame
    send_end_sequence(udp_socket, frame_number, uuid, frame_size, combined_checksums, server_address, False)
    # logging.info('Successfully sent frame to server') 

    # Lắng nghe phản hồi từ server qua cùng socket
    start_time = time.time()
    udp_socket.settimeout(5.0)  # Timeout chờ phản hồi
    while time.time() - start_time < 5.0:
        try:
            data, addr = udp_socket.recvfrom(1024)
            if data.startswith(b"FAILED:"):
                failed_sequences = list(map(int, data.decode().split(":")[1].split(",")))
                if not resend_failed_chunks(udp_socket, frame_number, uuid, frame_size, failed_sequences, sent_chunks, combined_checksums, server_address):
                    logging.error("Error during resend. Preparing for next frame.")
                    sent_chunks.clear()
                    return True  # Lỗi, chuẩn bị gửi frame mới
            elif data == b"SUCCESS":
                sent_chunks.clear()
                return True  # Thành công, chuẩn bị gửi frame mới
        except socket.timeout:
            break

    logging.error("Failed to send frame after retries.")
    sent_chunks.clear()
    return False  # Quá thời gian chờ, không thành công


def resend_failed_chunks(udp_socket, frame_number, uuid, frame_size, failed_sequences, sent_chunks, combined_checksums, server_address):
    """
    Resend các chunk bị thiếu trong thread riêng. Báo lỗi nếu sequence không tồn tại.
    """
    threads = []
    for sequence in failed_sequences:
        if sequence in sent_chunks:
            message = sent_chunks[sequence]
            udp_socket.sendto(message, server_address)
        else:
            logging.error(f"Missing sequence {sequence}. Cannot resend.")
            send_end_sequence(udp_socket, frame_number, uuid, frame_size, combined_checksums, server_address, True)
            sent_chunks.clear()
            return False  # Trả về trạng thái lỗi để vòng lặp `while` xử lý

    # Gửi cờ END để báo kết thúc resend nếu không có lỗi
    send_end_sequence(udp_socket, frame_number, uuid, frame_size, combined_checksums, server_address, False)
    logging.info("Sent end flag after resending failed chunks.")
    return True  # Thành công


def adjust_chunk_size(frame_size):
    """
    Điều chỉnh kích thước chunk nếu tổng số sequence vượt quá 65535.
    """
    global CHUNK_SIZE
    max_chunks = 65535  # Giới hạn tối đa của sequence (2 bytes)

    CHUNK_SIZE = max(frame_size // max_chunks, MIN_CHUNK_SIZE) # MIN_CHUNK_SIZE = 4096 => frame_size > 256MB thi update


    # Nếu tổng số chunk vượt quá 65535, tăng CHUNK_SIZE
    if CHUNK_SIZE != MIN_CHUNK_SIZE:
        logging.info(f"CHUNK_SIZE changed to {CHUNK_SIZE} to fit frame within sequence limit.")

async def send_frame_queue(udp_socket, server_address, frame_queue):
    """
    Lấy frame từ hàng đợi và gửi qua UDP.
    """
    frame_number = 0
    sent_frames = 0
    start_time = time.time()

    while True:
        try:
            # Lấy frame từ frame_queue với timeout 
            # frame = await asyncio.get_event_loop().run_in_executor(None, frame_queue.get, True, 0.2) 
            frame = await asyncio.wait_for(frame_queue.get(), timeout=0.2)
            frame_number += 1
            if frame_number > 9:
                frame_number = 1

            # Gửi frame qua UDP (sử dụng thread pool cho sync function)
            success = await asyncio.to_thread(send_frame, frame_number, udp_socket, frame, server_address)
            if success:
                frame_queue.task_done()
                sent_frames += 1

            # Tính toán FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = sent_frames / elapsed_time
                start_time = current_time
                sent_frames = 0
                logging.info(f"FPS: {fps:.2f}")

        except asyncio.CancelledError:
            logging.info("Sender task cancelled.")
            break
        except queue.Empty:
            logging.warning("Frame queue is empty. Pausing send frame...")
            await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Error during frame sending: {e}")
            await asyncio.sleep(1)

async def put_frame_queue(frame_queue, stop_event, camerahandler: CameraHandler):
    """
    Xử lý và đẩy frame từ camera vào queue dưới dạng async.
    
    Args: 
        frame_queue: Queue để chứa các frame
        stop_event: Event để dừng xử lý
        camerahandler: Instance của CameraHandler để chụp ảnh
    """
    frame_count = 0
    frame_number = 1  # Thêm biến đếm số frame
    start_time = asyncio.get_event_loop().time()

    # Tạo thư mục gốc để lưu frames nếu chưa tồn tại
    base_dir = "captured_frames"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    while not stop_event.is_set():
        try:
            # Kiểm tra queue đầy
            if frame_queue.full():
                logging.warning("Frame queue is full. Pausing process frame...")
                await asyncio.sleep(0.1)
                continue

            # Chụp ảnh từ camera 
            frame = camerahandler.capture_main_frame()
            if frame is None:
                logging.error("Failed to capture frame from camera")
                await asyncio.sleep(0.1)
                continue

            # Log kích thước frame gốc
            original_height, original_width = frame.shape[:2]
            original_size = frame.nbytes
            logging.info(f"Original frame size: {original_width}x{original_height}, {original_size/1024:.2f}KB")

            # Encode frame thành bytes với định dạng JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) 
            frame_bytes = encoded_frame.tobytes()
            
            # Log kích thước sau khi nén
            compressed_size = len(frame_bytes)
            logging.info(f"Compressed frame size: {compressed_size/1024:.2f}KB, Compression ratio: {original_size/compressed_size:.2f}x")

            # Đẩy frame vào queue
            try:
                await asyncio.wait_for(frame_queue.put(frame_bytes), timeout=0.5)
                frame_count += 1
                
                # Tính và log FPS mỗi giây 
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1:
                    fps = frame_count / elapsed_time
                    logging.info(f"Frame Capture FPS: {fps:.2f}")
                    start_time = current_time
                    frame_count = 0
                    
            except asyncio.TimeoutError:
                logging.warning("Timeout when putting frame to queue")
                continue

            # Đợi một chút để giảm tải CPU 
            await asyncio.sleep(0.01)

            # Tạo thư mục cho frame hiện tại
            frame_dir = os.path.join(base_dir, f"frame{frame_number}")
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)

            # Xử lý từng bounding box
            for i, box in enumerate(mask_boxes):
                try:
                    logging.debug(f"Xử lý bounding box {i+1}/{len(mask_boxes)}")
                    # Tính toán tọa độ 
                    x1, y1, x2, y2 = [int(coord / scale) for coord, scale in zip(box, [resize_ratio, resize_ratio, resize_ratio, resize_ratio])]
                    # x1, y1, x2, y2 = box
                    x1 = max(0, min(x1, original_width - 1))
                    y1 = max(0, min(y1, original_height - 1))
                    x2 = max(0, min(x2, original_width - 1))
                    y2 = max(0, min(y2, original_height - 1))

                    if x2 <= x1 or y2 <= y1:
                        logging.warning(f"Invalid bounding box: {x1}, {y1}, {x2}, {y2}. Skipping.")
                        continue

                    # Cắt và encode human_box
                    human_box = frame[y1:y2, x1:x2]
                    
                    # Log kích thước frame gốc
                    box_height, box_width = human_box.shape[:2] 
                    original_size = human_box.nbytes
                    logging.info(f"Original human box size: {box_width}x{box_height}, {original_size/1024:.2f}KB")
                    
                    # Encode frame
                    _, encoded_box = cv2.imencode('.jpg', human_box, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    encoded_bytes = encoded_box.tobytes()
                    
                    # Lưu human_box vào file
                    box_filename = os.path.join(frame_dir, f"box{i+1}.jpg")
                    cv2.imwrite(box_filename, human_box)
                    logging.info(f"Đã lưu {box_filename}")
                    
                except Exception as e:
                    logging.error(f"Lỗi xử lý bounding box: {e}", exc_info=True)
                    continue

            frame_count += 1
            frame_number += 1  # Tăng số frame sau khi xử lý xong
            current_time = asyncio.get_event_loop().time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                logging.info(f"FPS processing: {fps:.2f}")
                start_time = current_time
                frame_count = 0

        except Exception as e:
            logging.error(f"Error in put_frame_queue: {e}", exc_info=True) 
            await asyncio.sleep(0.1)

import os
import glob

async def put_frame_queue_2(frame_queue, stop_event, data_folder="data"):
    """
    Xử lý và đẩy frame từ thư mục data vào queue dưới dạng async.
    
    Args:
        frame_queue: Queue để chứa các frame
        stop_event: Event để dừng xử lý
        data_folder: Đường dẫn đến thư mục chứa các frame (mặc định là "data")
    """
    frame_count = 0
    start_time = asyncio.get_event_loop().time()
    
    # Lấy danh sách tất cả các file ảnh trong thư mục
    image_files = sorted(glob.glob(os.path.join(data_folder, "frame_*.png")))
    if not image_files:
        logging.error(f"Không tìm thấy file ảnh nào trong thư mục {data_folder}")
        return
        
    current_frame_idx = 0
    
    while not stop_event.is_set():
        try:
            # Kiểm tra queue đầy
            if frame_queue.full():
                logging.warning("Frame queue is full. Pausing process frame...")
                await asyncio.sleep(0.1)
                continue

            # Đọc frame từ file
            frame_path = image_files[current_frame_idx]
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.error(f"Không thể đọc frame từ {frame_path}")
                await asyncio.sleep(0.1)
                continue

            # Log kích thước frame gốc
            original_height, original_width = frame.shape[:2]
            original_size = frame.nbytes
            logging.info(f"Original frame size: {original_width}x{original_height}, {original_size/1024:.2f}KB")

            # Encode frame thành bytes với định dạng JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = encoded_frame.tobytes()
            
            # Log kích thước sau khi nén
            compressed_size = len(frame_bytes)
            logging.info(f"Compressed frame size: {compressed_size/1024:.2f}KB, Compression ratio: {original_size/compressed_size:.2f}x")

            # Đẩy frame vào queue
            try:
                await asyncio.wait_for(frame_queue.put(frame_bytes), timeout=0.5)
                frame_count += 1
                
                # Tính và log FPS mỗi giây
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1:
                    fps = frame_count / elapsed_time
                    logging.info(f"Frame Capture FPS: {fps:.2f}")
                    start_time = current_time
                    frame_count = 0
                    
            except asyncio.TimeoutError:
                logging.warning("Timeout when putting frame to queue")
                continue

            # Chuyển sang frame tiếp theo, nếu hết thì quay lại frame đầu
            current_frame_idx = (current_frame_idx + 1) % len(image_files)

            # Đợi một chút để giảm tải CPU
            await asyncio.sleep(0.01)

        except Exception as e:
            logging.error(f"Error in put_frame_queue_2: {e}", exc_info=True)
            await asyncio.sleep(0.1)

async def process_frames_queue(camerahandler: CameraHandler, background_remover: BackgroundRemover, frame_queue: asyncio.Queue, stop_event: asyncio.Event):
    """Xử lý khung hình và đẩy vào hàng đợi một cách bất đồng bộ."""  
    prev_frame = None
    frame_count = 0
    frame_number = 1  # Thêm biến đếm số frame
    start_time = asyncio.get_event_loop().time()

    # Tạo thư mục gốc để lưu frames nếu chưa tồn tại
    base_dir = "captured_frames"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    while not stop_event.is_set():
        try:
            # Kiểm tra queue đầy
            if frame_queue.full():
                logging.warning("Frame queue is full. Pausing frame processing...")
                await asyncio.sleep(0.2)
                continue

            # Lấy original_frame từ capture_lores_frame
            lores_frame = camerahandler.capture_lores_frame()
            if lores_frame is None:
                logging.error("Không thể chụp ảnh lores từ camera.")
                await asyncio.sleep(0.1)
                continue

            logging.debug("Bắt đầu xử lý background removal...")
            # Xử lý background
            _, foreground, mask_boxes = background_remover.remove_background(lores_frame)
            logging.debug(f"Đã tìm thấy {len(mask_boxes)} bounding boxes")

            # Kiểm tra thay đổi so với prev_frame
            if prev_frame is not None and cv2.absdiff(foreground, prev_frame).sum() < 10000:
                logging.debug("Không có thay đổi đáng kể, bỏ qua frame")
                await asyncio.sleep(0.1)
                continue

            # Cập nhật prev_frame
            prev_frame = foreground.copy()
            
            logging.debug("Bắt đầu chụp ảnh main...")
            # main_frame = camerahandler.capture_main_frame()
            # if main_frame is None:
            #     logging.error("Không thể chụp ảnh main từ camera.")
            #     await asyncio.sleep(0.1)
            #     continue

            original_height, original_width = lores_frame.shape[:2]
            logging.info(f"original_frame size (width x height): ({original_width}x{original_height})")

            # # Lấy x_scale và y_scale
            # x_scale = camerahandler.x_scale 
            # y_scale = camerahandler.y_scale 
            resize_ratio = background_remover.resize_ratio

            # Tạo thư mục frame nếu chưa tồn tại
            frame_dir = os.path.join(base_dir, f"frame{frame_number}")
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)

            # Lưu human_box vào file
            foreground_filename = os.path.join(frame_dir, f"foreground.jpg")
            cv2.imwrite(foreground_filename, foreground)
                        
            # Xử lý từng bounding box
            for i, box in enumerate(mask_boxes):
                try:
                    logging.debug(f"Xử lý bounding box {i+1}/{len(mask_boxes)}")
                    # Tính toán tọa độ 
                    x1, y1, x2, y2 = [int(coord / scale) for coord, scale in zip(box, [resize_ratio, resize_ratio, resize_ratio, resize_ratio])]
                    # x1, y1, x2, y2 = box
                    x1 = max(0, min(x1, original_width - 1))
                    y1 = max(0, min(y1, original_height - 1))
                    x2 = max(0, min(x2, original_width - 1))
                    y2 = max(0, min(y2, original_height - 1))

                    if x2 <= x1 or y2 <= y1:
                        logging.warning(f"Invalid bounding box: {x1}, {y1}, {x2}, {y2}. Skipping.")
                        continue

                    # Cắt và encode human_box 
                    human_box = lores_frame[y1:y2, x1:x2]
                    
                    # Log kích thước frame gốc 
                    box_height, box_width = human_box.shape[:2] 
                    original_size = human_box.nbytes
                    logging.info(f"Original human box size: {box_width}x{box_height}, {original_size/1024:.2f}KB")                 
                        
                    # Lưu human_box vào file
                    box_filename = os.path.join(frame_dir, f"box{i+1}.jpg")
                    cv2.imwrite(box_filename, human_box)
                    logging.info(f"Đã lưu {box_filename}")
                    
                except Exception as e:
                    logging.error(f"Lỗi xử lý bounding box: {e}", exc_info=True)
                    continue

            frame_count += 1
            frame_number += 1  # Tăng số frame sau khi xử lý xong
            current_time = asyncio.get_event_loop().time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                logging.info(f"FPS processing: {fps:.2f}")
                start_time = current_time
                frame_count = 0

        except Exception as e:
            logging.error(f"Lỗi trong process_frames_queue: {e}", exc_info=True)
            if prev_frame is not None:
                del prev_frame
            await asyncio.sleep(0.1)
            continue

async def start_sender(frame_queue):
    server_address = (SERVER_IP, SERVER_PORT)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setblocking(True)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    try:
        log_cpu_temp()
        await send_frame_queue(udp_socket, server_address, frame_queue)
    except asyncio.CancelledError:
        logging.info("Sender task cancelled.")
    finally:
        udp_socket.close()
        logging.info("Sender stopped.")

async def main():
    frame_queue = asyncio.Queue(maxsize=1000)
    stop_event = asyncio.Event()
    
    # Khởi tạo UDP socket 
    server_address = (SERVER_IP, SERVER_PORT)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setblocking(True)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    try:
        # Khởi tạo background remover 
        background_remover.learn_background(camerahandler)
        log_cpu_temp()

        # Tạo các task riêng biệt 
        processor_task = asyncio.create_task(
            process_frames_queue(camerahandler, background_remover, frame_queue, stop_event)
            # put_frame_queue(frame_queue, stop_event, camerahandler)
            # put_frame_queue_2(frame_queue, stop_event)
        )    

        # Chờ cả hai task hoàn thành
        await asyncio.gather(processor_task)

    except asyncio.CancelledError:
        logging.info("Main task cancelled.")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
    finally:
        # Dọn dẹp
        stop_event.set()
        udp_socket.close()
        
        # Chờ các task kết thúc  
        if 'processor_task' in locals():
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
        logging.info("Main stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program stopped by user.")