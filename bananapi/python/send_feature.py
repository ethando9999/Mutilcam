from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import socket
import zlib
import time
import queue
import logging
import os
import platform
import asyncio

from object_detection.yolo_pose import HumanDetection
from body_color.pose_cluster import PoseClusterProcessor

import orjson
import cv2
import numpy as np

from logging_python_bananapi import setup_logging, get_logger, log_fps

# Thiết lập logging 
setup_logging()
logger = get_logger(__name__)

# Cấu hình kết nối
SERVER_IP = "192.168.1.123" 
SERVER_PORT = 5050 
MIN_CHUNK_SIZE = 4096
CHUNK_SIZE = MIN_CHUNK_SIZE
PASSWORD = "secret_password"
AES_KEY = sha256(PASSWORD.encode()).digest() 
INIT_FLAG = 1
END_FLAG = 2
ERROR_FLAG = 3
FEATURE_FLAG = 4

MAX_RETRY = 3  # Giới hạn số lần retry

# Tạo IV dùng chung cho cả ứng dụng
iv = get_random_bytes(16)

detector = HumanDetection()
pose_processor = PoseClusterProcessor()

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


def send_init_sequence(udp_socket, frame_number, uuid, chunk_size, total_chunks, server_address):
    """
    Gửi gói tin "init" để bắt đầu gửi frame.
    """
    # logging.info(f"Start init sequence with UUID {uuid} to server.")
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

def send_frame(frame_number, udp_socket, frame, uuid, server_address):
    """
    Gửi một frame qua UDP và nhận phản hồi trực tiếp từ server.
    """
    sent_chunks = {}  # Đảm bảo sent_chunks được khởi tạo trong scope của hàm
    try:
        frame_size = len(frame)
        adjust_chunk_size(frame_size)
        total_chunks = (frame_size + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Gửi gói tin init
        if not send_init_sequence(udp_socket, frame_number, uuid, CHUNK_SIZE, total_chunks, server_address):
            return False

        combined_checksums = 0

        # Gửi từng chunk
        for sequence in range(1, total_chunks + 1):
            start_idx = (sequence - 1) * CHUNK_SIZE
            chunk = frame[start_idx:start_idx + CHUNK_SIZE]
            
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

        # Gửi END sequence
        send_end_sequence(udp_socket, frame_number, uuid, frame_size, combined_checksums, server_address)

        # Chờ phản hồi
        udp_socket.settimeout(5.0)
        start_time = time.time()
        while time.time() - start_time < 5.0:
            try:
                data, addr = udp_socket.recvfrom(1024)
                if data == b"SUCCESS":
                    return True
                elif data.startswith(b"FAILED:"):
                    failed_sequences = list(map(int, data.decode().split(":")[1].split(",")))
                    if not resend_failed_chunks(udp_socket, frame_number, uuid, frame_size, 
                                             failed_sequences, sent_chunks, combined_checksums, server_address):
                        return False
            except socket.timeout:
                break

    except Exception as e:
        logging.error(f"Error sending frame: {e}")
    finally:
        sent_chunks.clear()
    
    return False

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

async def send_frame_queue(udp_socket, server_address, processed_frame_queue):
    frame_number = 0
    sent_frames = 0
    last_fps_time = time.time()
    
    while True:
        try:
            # Lấy và xử lý frame ngay khi có
            processed_data = await processed_frame_queue.get()
            
            frame_number = (frame_number % 9) + 1
            
            # Lấy dữ liệu từ processed_data
            frame_bytes = processed_data["frame"]
            frame_uuid = processed_data["uuid"]
            feature_data_bytes = processed_data["feature_data"]
            
            # Gửi frame
            success = await asyncio.to_thread(
                send_frame, frame_number, udp_socket, 
                frame_bytes, frame_uuid, 
                server_address
            )
            
            # Gửi feature data nếu gửi frame thành công
            if success:
                feature_success = await send_feature(
                    udp_socket, frame_number, frame_uuid,
                    feature_data_bytes, server_address
                )
                
                if feature_success:
                    sent_frames += 1
                    processed_frame_queue.task_done()
            else:
                logger.error(f"Failed to send frame {frame_number}")

            # Tính và log FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1:
                fps = sent_frames / (current_time - last_fps_time)
                log_fps(fps, __name__)
                sent_frames = 0
                last_fps_time = current_time

        except Exception as e:
            logger.error(f"Error in send_frame_queue: {e}")
            await asyncio.sleep(0.1)

async def send_feature(udp_socket, frame_number, uuid, feature_data_bytes, server_address):
    """
    Gửi feature data đã được serialize qua UDP với cờ FEATURE_FLAG.
    
    Args:
        udp_socket: Socket UDP để gửi dữ liệu
        frame_number: Số thứ tự frame
        uuid: UUID của frame
        feature_data_bytes: Dữ liệu feature đã được serialize
        server_address: Địa chỉ server đích
    """
    try:
        sequence_number = 0  # Sequence number cho gói tin feature

        # Tạo payload với cờ FEATURE_FLAG
        payload = FEATURE_FLAG.to_bytes(1, 'big') + uuid.encode() + feature_data_bytes

        # Tính checksum
        checksum = calculate_checksum(frame_number, sequence_number, payload, uuid)

        # Tạo header
        header = (
            frame_number.to_bytes(1, 'big') +
            sequence_number.to_bytes(2, 'big') +
            checksum.to_bytes(2, 'big')
        )

        # Kết hợp header và payload
        message = header + payload

        # Gửi qua UDP
        udp_socket.sendto(message, server_address)
        logger.info(f"Sent feature data for frame UUID: {uuid}")

        # Chờ xác nhận từ server
        udp_socket.settimeout(1.0)
        try:
            data, addr = udp_socket.recvfrom(1024)
            if data == b"END":
                logger.info("Received END from server!")
                return True
        except socket.timeout:
            logger.error("Timeout waiting for server acknowledgment for feature data.")
            return False

    except Exception as e:
        logger.error(f"Failed to send feature data: {e}")
        return False
    

async def start_sender(frame_queue, include_put_frame=False, num_workers=4):
    """
    Khởi động sender để gửi frame qua UDP socket.
    
    Args:
        frame_queue: Queue chứa các frame đã được xử lý
        include_put_frame: Không còn sử dụng
        num_workers: Không còn sử dụng
    """
    server_address = (SERVER_IP, SERVER_PORT)
    udp_socket = None

    try:
        # Khởi tạo UDP socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setblocking(True)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

        log_cpu_temp()
        logger.info("Starting sender...")
        
        # Chỉ chạy task gửi frame
        await send_frame_queue(udp_socket, server_address, frame_queue)
                
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Sender task cancelled.")
    except Exception as e:
        logger.error(f"Error in start_sender: {e}", exc_info=True)
    finally:
        # Dọn dẹp tài nguyên
        if udp_socket:
            udp_socket.close()
        logger.info("Sender stopped.")

async def main():
    """Entry point khi chạy trực tiếp file này."""
    try:
        frame_queue = asyncio.Queue(maxsize=2000)
        await start_sender(frame_queue, include_put_frame=True, num_workers=4)
    except KeyboardInterrupt:
        logger.info("Program stopped by user.")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program stopped by user.")