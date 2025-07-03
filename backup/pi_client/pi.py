from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import socket
import zlib
import threading
import time
import queue
import random
import logging
import os
import platform

# Cấu hình kết nối
SERVER_IP = "192.168.1.100"
SERVER_PORT = 9090
MIN_CHUNK_SIZE = 4096
CHUNK_SIZE = MIN_CHUNK_SIZE
PASSWORD = "secret_password"
# AES_KEY = PASSWORD.encode().ljust(32, b'\0')[:32]  # AES key: 32 bytes
AES_KEY = sha256(PASSWORD.encode()).digest()
INIT_FLAG = 1
END_FLAG = 2
ERROR_FLAG = 3

MAX_RETRY = 3  # Giới hạn số lần retry

frame_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()

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

# def aes_encrypt(data):
#     iv = get_random_bytes(16)
#     cipher = AES.new(AES_KEY, AES.MODE_CBC, iv=iv)
#     return iv + cipher.encrypt(pad(data, AES.block_size))
def aes_encrypt(data):
    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv=iv)
    
    # Kiểm tra kích thước dữ liệu
    # logging.debug(f"Data length before padding: {len(data)}")
    padded_data = pad(data, AES.block_size)
    # logging.debug(f"Padded data length: {len(padded_data)}")
    
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


def send_frame_queue(udp_socket, server_address):
    """Lấy frame từ hàng đợi và gửi qua UDP."""

    frame_number = 0

    sent_frames = 0
    start_time = time.time()

    while not stop_event.is_set():
        try:
            # nếu đang ko có frame, tiến hành sleep 1 
            if frame_queue.empty():
                logging.warning(f"Frame queue is empty. Pausing send frame...")
                time.sleep(1)

            # tinh toan frame_number
            frame = frame_queue.get(timeout=0.2)  # Lấy frame từ hàng đợi
            frame_number += 1
            if frame_number > 9:
                frame_number = 1

            # Retry logic
            retry_count = 0
            # while retry_count < MAX_RETRY:
                # logging.info(f"Sending frame {frame_number} with count {retry_count} retries
            if send_frame(frame_number, udp_socket, frame, server_address):
                frame = None
                frame_queue.task_done()  # Đánh dấu frame đã được xử lý
                sent_frames += 1

                # Tính toán FPS
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1:
                    fps = sent_frames / elapsed_time
                    start_time = current_time
                    sent_frames = 0
                    logging.info(f"fps {fps}.")
                #     break  # Thành công, tiếp tục frame tiếp theo
                # retry_count += 1
                # logging.warning(f"Retrying frame {frame_number}, attempt {retry_count}/{MAX_RETRY}.")
                # time.sleep(5)
            # else:
            #     # Nếu vượt quá số lần retry, bỏ qua frame này
            #     logging.error(f"Max retries reached for frame {frame_number}. Skipping frame.")

                
        except queue.Empty:
            time.sleep(1)
            continue
        except Exception as e:
            logging.error(f"Error during frame sending: {e}")
            break

def process_frame_queue():

    # frame_queue.put(b"A" * (1024 * 1024))  # Frame 1MB
    # return
    filename = "pi_client/file.webp"

    # Open the file in binary read mode
    with open(filename, "rb") as file: 
        # Read the file content as bytes
        byte_content = file.read() 

    while not stop_event.is_set(): 
        if frame_queue.full():
            logging.warning("Frame queue is full. Pausing process frame...") 
            # #debug
            time.sleep(0.1) 
            # time.sleep(10)
            continue
        try:
            # Đẩy dữ liệu frame giả lập vào hàng đợi để gửi
            for i in range(30):
                # frame_queue.put(b"A" * (500 * 1024))  # Frame 500 KB
                frame_queue.put(byte_content)  # Frame 500 KB
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error during frame processing: {e}")
            time.sleep(1)
            continue

def main():

    server_address = (SERVER_IP, SERVER_PORT)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    udp_socket.setblocking(True)
    # udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 12096)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)


    try:
        logging.basicConfig(level=logging.DEBUG)
        log_cpu_temp()

        logging.info("Starting processing and transmission.")

        processing_thread = threading.Thread(
            target=process_frame_queue, args=(), daemon=True)
        sending_thread = threading.Thread(
            target=send_frame_queue, args=(udp_socket, server_address), daemon=True)

        processing_thread.start()
        sending_thread.start()

        processing_thread.join()
        sending_thread.join()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        stop_event.set()  
        udp_socket.close()
        logging.info("Resources cleaned up. Exiting.")




if __name__ == "__main__":
    main()
