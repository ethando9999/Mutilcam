import socket
import threading
import queue
import zlib
import cv2
import time
import os
from utils.rmbg_mog import BackgroundRemover
from utils.camera_log import logging
from utils.setup_camera import CameraHandler

# Cấu hình kết nối
SERVER_DOMAIN = "center.pi" 
SERVER_PORT = 9090 
CHUNK_SIZE = 2048
END_CHUNK = b"END"
MAX_RETRY = 3
LOCAL_UDP_PORT = 9091  # UDP server port for receiving resend requests from GoLang

SERVER_IP = None  # Tự động gán từ SERVER_DOMAIN ở hàm main

lock = threading.Lock()
frame_queue = queue.Queue(maxsize=20)  # Hàng đợi để lưu frame đã xử lý
stop_event = threading.Event() 

# Event to signal success confirmation

success_event = threading.Event()

def resolve_domain_to_ip(domain):
    """Phân giải domain thành địa chỉ IP."""
    try:
        ip_address = socket.gethostbyname(domain)
        logging.info(f"Domain '{domain}' resolved to IP: {ip_address}")
        return ip_address
    except socket.gaierror as e:
        logging.error(f"Failed to resolve domain '{domain}': {e}")
        return None

def calculate_checksum(data): 
    """Tính checksum 4 byte của dữ liệu."""
    return zlib.crc32(data)


# def udp_resend_server(sent_chunks):
#     """Khởi động một UDP server để nhận yêu cầu resend từ GoLang."""
#     udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     udp_server_socket.bind(("0.0.0.0", LOCAL_UDP_PORT))
#     logging.info(f"UDP server started on port {LOCAL_UDP_PORT}.")

#     def resend_chunk(sequence, addr):
#         """Gửi lại một chunk bất đồng bộ."""
#         try:
#             if sequence in sent_chunks:
#                 udp_server_socket.sendto(sent_chunks[sequence], addr)
#                 logging.info(f"Resent sequence {sequence} to {addr}.")
#             else:
#                 logging.warning(f"Sequence {sequence} not found in sent_chunks.") 
#         except Exception as e:
#             logging.error(f"Error resending sequence {sequence}: {e}")

#     while not stop_event.is_set():
#         try:
#             udp_server_socket.settimeout(1.0)  # Timeout để không treo server
#             data, addr = udp_server_socket.recvfrom(1024)  # Nhận dữ liệu từ GoLang
#             logging.info(f"Received message from {addr}: {data}")

#             if data.startswith(b"FAILED:"): 
#                 # Phân tích danh sách sequence lỗi từ GoLang
#                 failed_sequences = list(map(int, data.decode().split(":")[1].split(",")))
#                 logging.info(f"Failed sequences received: {failed_sequences}")

#                 # Gửi lại các chunk tương ứng bất đồng bộ
#                 for sequence in failed_sequences:
#                     threading.Thread(target=resend_chunk, args=(sequence, addr), daemon=True).start()

#                 send_end_sequence(udp_server_socket, addr)

#             elif data == b"SUCCESS":
#                 logging.info("Received SUCCESS message from server.")
#                 success_event.set()  # Signal success
#         except socket.timeout:
#             continue 
#         except Exception as e:
#             logging.error(f"Error in UDP resend server: {e}") 

#     udp_server_socket.close()
#     logging.info("UDP resend server stopped.")

def udp_resend_server(sent_chunks):
    """Khởi động một UDP server để nhận yêu cầu resend từ GoLang."""
    udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server_socket.bind(("0.0.0.0", LOCAL_UDP_PORT))
    logging.info(f"UDP server started on port {LOCAL_UDP_PORT}.")

    while not stop_event.is_set():
        try:
            udp_server_socket.settimeout(1.0)  # Timeout để không treo server
            data, addr = udp_server_socket.recvfrom(1024)  # Nhận dữ liệu từ GoLang
            logging.info(f"Received message from {addr}: {data}")

            if data.startswith(b"FAILED:"): 
                # Phân tích danh sách sequence lỗi từ GoLang
                failed_sequences = list(map(int, data.decode().split(":")[1].split(",")))
                logging.info(f"Failed sequences received: {failed_sequences}")

                total_packets = 0  # Đếm tổng số gói tin
                # Gửi lại các chunk tương ứng
                for sequence in failed_sequences:
                    if sequence in sent_chunks:                        
                        udp_server_socket.sendto(sent_chunks[sequence], addr)   
                        total_packets += 1  # Tăng số gói tin
                        logging.info(f"Resent sequence {sequence} to {addr}.")
                    else:
                        logging.warning(f"Sequence {sequence} not found in sent_chunks.")
                send_end_sequence(udp_server_socket, addr, total_packets)
                
            elif data == b"SUCCESS":
                logging.info("Received SUCCESS message from server.")
                success_event.set()  # Signal success
        except socket.timeout:
            continue 
        except Exception as e:
            logging.error(f"Error in UDP resend server: {e}") 

    udp_server_socket.close()
    logging.info("UDP resend server stopped.")

def send_end_sequence(udp_socket, server_address, total_packets):

    """Gửi chuỗi kết thúc cùng với tổng số gói tin."""
    try:
        sequence_number = 0
        checksum = 0 
        sequence = sequence_number.to_bytes(4, 'big')
        checksum_bytes = checksum.to_bytes(4, 'big')
        total_packets_bytes = total_packets.to_bytes(4, 'big')  # Tổng gói tin (4 byte)
        message = sequence + checksum_bytes + total_packets_bytes
        udp_socket.sendto(message, server_address)
        logging.info(f"Sent end sequence with total packets: {total_packets}")
    except socket.error as e:
        logging.error(f"Error sending end sequence: {e}")

def send_frame(udp_socket, frame, server_address, sent_chunks):
    """Gửi frame qua UDP."""
    try:
        frame_size = len(frame)
        sequence_number = 1
        total_packets = 0  # Đếm tổng số gói tin

        for i in range(0, frame_size, CHUNK_SIZE):
            chunk = frame[i:i + CHUNK_SIZE]   
            checksum = calculate_checksum(chunk)
            sequence = sequence_number.to_bytes(4, 'big')
            checksum_bytes = checksum.to_bytes(4, 'big')
            message = sequence + checksum_bytes + chunk 
 
            udp_socket.sendto(message, server_address)  
            sent_chunks[sequence_number] = message  # Lưu lại chunk đã gửi
            sequence_number += 1 
            total_packets += 1  
 
        # Gửi chuỗi kết thúc cùng với tổng số gói tin 
        send_end_sequence(udp_socket, server_address, total_packets) 
        
        start_time = time.time() 

        while not success_event.is_set():
            if time.time() - start_time > 10:  # Timeout sau 10 giây
                logging.warning("No success confirmation received within the timeout period.")
                return False  
            time.sleep(0.1)  # Tránh tốn tài nguyên CPU

        logging.info("Frame transmission confirmed as successful.")    
        return True 
    
    except socket.error as e: 
        logging.error(f"Transmission error: {e}") 
        return False    

def retry_send(udp_socket, frame, server_address, sent_chunks):
    """Thử gửi lại frame nếu có lỗi."""
    for attempt in range(MAX_RETRY):
        if send_frame(udp_socket, frame, server_address, sent_chunks):
            return True
        logging.warning(f"Retrying frame transmission {attempt + 1}/{MAX_RETRY}...")
    logging.error("Maximum retries reached. Frame transmission failed.")
    return False


def send_frames(udp_socket, server_address):
    """Lấy frame từ hàng đợi và gửi qua UDP."""
    sent_chunks = {}  # Lưu các chunk đã gửi để xử lý resend
 
    # Khởi chạy luồng UDP server lắng nghe resend requests
    resend_server_thread = threading.Thread(
        target=udp_resend_server, args=(sent_chunks,), daemon=True)
    resend_server_thread.start()

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.2)  # Lấy frame từ hàng đợi
            if not retry_send(udp_socket, frame, server_address, sent_chunks):
                logging.error("Failed to send frame after retries. Exiting.")
                break 
        except queue.Empty: 
            continue
        except Exception as e:
            logging.error(f"Error during frame sending: {e}")
            break

def learn_background(camerahandler: CameraHandler, background_remover: BackgroundRemover, learning_time=3):
    """
    Learn the background for background removal.

    Args:
        - output: Stream of frames from the camera.
        - background_remover: An instance of BackgroundRemover to update the background model.
        - learning_time (float): Duration for learning the background (in seconds).
    """
    logging.info("Starting background learning...")
    start_time = time.time()

    while time.time() - start_time <= learning_time:
        try:
            frame = camerahandler.picam2.capture_array()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background_remover.update_background(gray_frame)
            # time.sleep(0.1)  # Tạm dừng một khoảng thời gian ngắn, để giảm tải số lượng hình cần phải xử lý cho CPU
        except Exception as e:
            logging.error(f"An error occurred during background learning: {e}")
    logging.info("Background learning completed successfully!")

def process_frames(camerahandler: CameraHandler, background_remover: BackgroundRemover):
    """Xử lý khung hình và đẩy vào hàng đợi."""
    prev_frame = None
    frame_count = 0
    switch_frame_count = 0  # Bộ đếm khung hình để kiểm soát chuyển đổi cấu hình
    x_scale, y_scale = None, None 
    start_time = time.time()  # Khởi tạo biến thời gian  

    while not stop_event.is_set():
        if frame_queue.full():
            logging.warning("Frame queue is full. Pausing frame processing...")
            time.sleep(0.2)
            continue

        try:
            original_frame = camerahandler.picam2.capture_array()
            gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            if x_scale is not None and y_scale is not None:
                gray_frame = cv2.resize(gray_frame, 
                                        (int(original_frame.shape[1] * x_scale), 
                                         int(original_frame.shape[0] * y_scale)))
            _, foreground, mask_boxes = background_remover.remove_background(gray_frame)

            # Kiểm tra thay đổi so với prev_frame
            if prev_frame is not None and cv2.absdiff(foreground, prev_frame).sum() < 10000: 
                if not camerahandler.is_default_config: # and switch_frame_count >= 50:
                    logging.info("Switching to lower resolution...")
                    camerahandler.default_config_camera()
                    switch_frame_count = 0
                else:
                    switch_frame_count += 1 
                continue

            # Xử lý chuyển đổi cấu hình
            if camerahandler.is_default_config: # and switch_frame_count >= 50:
                logging.info("Switching to max resolution...") 
                x_scale, y_scale = camerahandler.change_max_resolution()
                switch_frame_count = 0
                continue
            elif camerahandler.is_default_config:
                switch_frame_count += 1 

            # Cập nhật prev_frame
            prev_frame = foreground.copy() 

            original_height, original_width = original_frame.shape[:2]
            logging.info(f"original_frame size (width x height): ({original_width}x{original_height})")  

            # Xử lý bounding boxes 
            for box in mask_boxes:
                # logging.info(f"x_scale, y_scale: ({x_scale:.3f},{y_scale:.3f})")
                # x1, y1, x2, y2 = box
                # scale_box = gray_frame[y1:y2, x1:x2]
                # logging.info(f"detected box size on scale box: {scale_box.shape[:2]}")  

                x1, y1, x2, y2 = [int(coord / scale) for coord, scale in zip(box, [x_scale, y_scale, x_scale, y_scale])] 
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))  
                x2 = max(0, min(x2, original_width - 1))  
                y2 = max(0, min(y2, original_height - 1))

                if x2 <= x1 or y2 <= y1:
                    logging.warning(f"Invalid bounding box: {x1}, {y1}, {x2}, {y2}. Skipping.") 
                    continue

                human_box = original_frame[y1:y2, x1:x2]
                logging.info(f"Do phan giai frame gui di: {human_box.shape[:2]}")
                if len(mask_boxes) == 1:
                    del original_frame  # Giải phóng bộ nhớ nếu chỉ có 1 bounding box

                _, encoded_box = cv2.imencode('.jpg', human_box, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_queue.put(encoded_box.tobytes())

            frame_count += 1 
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                logging.info(f"FPS: {fps:.2f}")
                start_time = current_time
                frame_count = 0

        except Exception as e:
            logging.error(f"Error during frame processing: {e}")
            if prev_frame:
                del prev_frame 
            continue
 
def log_cpu_temp():
    """Ghi lại nhiệt độ CPU."""
    try:
        temp = os.popen("vcgencmd measure_temp").readline().strip()
        logging.info(f"CPU Temperature: {temp}")
    except Exception as e:
        logging.error(f"Error reading CPU temperature: {e}")

def main():
    global SERVER_IP
    SERVER_IP = resolve_domain_to_ip(SERVER_DOMAIN)
    if not SERVER_IP:
        logging.error("Failed to resolve server domain. Exiting.")
        exit(1)

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setblocking(True)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096) 
    server_address = (SERVER_IP, SERVER_PORT)

    background_remover = BackgroundRemover()
    camerahandler = CameraHandler()

    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    try:
        learn_background(camerahandler, background_remover)
        time.sleep(5)  # Allow CPU temp logging
        log_cpu_temp()

        logging.info("Starting processing and transmission.")

        processing_thread = threading.Thread(
            target=process_frames, args=(camerahandler, background_remover), daemon=True)
        sending_thread = threading.Thread(
            target=send_frames, args=(udp_socket, server_address), daemon=True)

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
