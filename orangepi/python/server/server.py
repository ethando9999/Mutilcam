import os
import socket
import threading
import queue as Queue  # Đổi tên module queue thành Queue để tránh xung đột
import time
import platform
import subprocess
import asyncio

import json
import struct
from utils.logging_python_orangepi import setup_logging, get_logger

import numpy as np
import cv2

# Thiết lập logging khi khởi động
setup_logging()
logger = get_logger(__name__)

SOCKET_PATH = "/mnt/ramdisk/ai_socket"  # Đường dẫn socket hợp lệ

def check_existing_ramdisk(ramdisk):
    """Kiểm tra xem đã có RAMDisk nào được mount chưa."""
    system_platform = platform.system()

    if system_platform == "Darwin":  # macOS 
        try:
            # Lệnh để lấy tất cả các ổ đĩa đã được mount
            mount_output = subprocess.check_output("mount", shell=True).decode("utf-8")
            # Kiểm tra xem có thư mục nào chứa "RAMDisk" không
            for line in mount_output.splitlines():
                if "RAMDisk" in line and ramdisk in line:
                    print(f"RAMDisk already mounted at {ramdisk}")
                    return True  # Đã có RAMDisk, không cần tạo mới
            return False  # Không có RAMDisk nào, cần tạo mới
        except subprocess.CalledProcessError as e:
            print(f"Error checking mount: {e}")
            return False
    elif system_platform == "Linux" or system_platform == "Linux2":  # Linux / Raspberry Pi
        try:
            # Lệnh để lấy tất cả các ổ đĩa đã được mount
            mount_output = subprocess.check_output("mount", shell=True).decode("utf-8")
            # Kiểm tra xem có thư mục nào chứa "tmpfs" (loại hệ thống file được sử dụng cho RAMDisk) không
            for line in mount_output.splitlines():
                if "tmpfs" in line and ramdisk in line:
                    print(f"RAMDisk already mounted at {ramdisk}")
                    return True  # Đã có RAMDisk, không cần tạo mới
            return False  # Không có RAMDisk nào, cần tạo mới
        except subprocess.CalledProcessError as e:
            print(f"Error checking mount: {e}")
            return False
    else:
        print("This script only works on macOS and Linux.")
        return False

def create_ramdisk(ramdisk): 
    """Tạo RAMDisk nếu chưa có."""
    system_platform = platform.system()

    if system_platform == "Darwin":  # macOS
        if not check_existing_ramdisk(ramdisk):
            print("Creating new RAMDisk on macOS...")
            ramdisk_size_in_blocks = 32768  # 16MB ÷ 512 bytes = 32768 blocks
            ramdisk_command = f"diskutil erasevolume HFS+ RAMDisk `hdiutil attach -nomount ram://{ramdisk_size_in_blocks}`"
            
            try:
                subprocess.check_call(ramdisk_command, shell=True)
                print(f"RAMDisk created and mounted at {ramdisk}.")
                return ramdisk
            except subprocess.CalledProcessError as e:
                print(f"Error creating RAMDisk: {e}")
                return None
        else:
            return ramdisk

    elif system_platform == "Linux" or system_platform == "Linux2":  # Linux / Raspberry Pi
        if not check_existing_ramdisk(ramdisk):
            print("Creating new RAMDisk on Linux/Raspberry Pi...")
            
            try:
                # Sử dụng sudo để tạo thư mục RAMDisk nếu chưa có
                subprocess.check_call(f"sudo mkdir -p {ramdisk}", shell=True)
                
                # Mount RAMDisk với kích thước 16MB
                subprocess.check_call(f"sudo mount -t tmpfs -o size=500M tmpfs {ramdisk}", shell=True)
                print(f"RAMDisk created and mounted at {ramdisk}.")
                return ramdisk
            except subprocess.CalledProcessError as e:
                print(f"Error creating RAMDisk: {e}")
                return None
        else:
            return ramdisk

    else:
        print("This script only works on macOS and Linux.")
        return None


def unmount_ramdisk():
    """Gỡ bỏ RAMDisk."""
    system_platform = platform.system()

    if system_platform == "Darwin":  # macOS
        print("Unmounting RAMDisk...")
        try:
            subprocess.check_call("diskutil unmount /Volumes/RAMDisk", shell=True)
            print("RAMDisk unmounted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error unmounting RAMDisk: {e}")

    elif system_platform == "Linux" or system_platform == "Linux2":  # Linux / Raspberry Pi
        print("Unmounting RAMDisk...")
        try:
            subprocess.check_call("sudo umount /ramdisk", shell=True)
            print("RAMDisk unmounted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error unmounting RAMDisk: {e}")

    else:
        print("This script only works on macOS and Linux.")

def decode_color_data(color_signature_data):
    """
    Giải mã dữ liệu màu từ định dạng dictionary về dữ liệu ban đầu.
    
    Parameters:
        color_signature_data (dict): Dictionary chứa dữ liệu màu đã được tuần tự hóa.
    
    Returns:
        list: Danh sách body_color_data được tái tạo với các giá trị np.ndarray và các giá trị None.
    """
    try:
        max_index = max(idx for idx, _ in color_signature_data) if color_signature_data else -1
        body_color_data = [None] * (max_index + 1)
        
        for idx, color in color_signature_data:
            body_color_data[idx] = np.array(color, dtype=np.uint8)
        
        return body_color_data
    except Exception as e:
        logger.error(f"Failed to decode color data: {e}")
        return []

async def decompress_frame(encoded_frame: bytes) -> np.ndarray:
    """
    Giải nén frame từ định dạng JPEG.

    Args:
        encoded_frame: Dữ liệu ảnh đã được nén (bytes)

    Returns:
        np.ndarray: Frame đã giải nén (numpy array)
    """
    try:
        nparr = np.frombuffer(encoded_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Giải nén thất bại, có thể dữ liệu bị lỗi.")
        return frame
    except Exception as e:
        logger.error(f"Lỗi khi giải nén frame: {e}")
        return None

async def worker(task_queue, frame_queue):
    while True:
        client_conn, client_id = await task_queue.get()
        if client_conn is None:
            break  # Kết thúc khi nhận tín hiệu dừng

        try:
            loop = asyncio.get_event_loop()
            
            # Đọc frame size và feature_data_size từ client 
            header = await loop.sock_recv(client_conn, 8)  # Đọc 8 byte header chứa frame_size và feature_data_size
            frame_size, feature_data_size = struct.unpack('>II', header)

            # Đọc frame
            frame_data = await receive_all(client_conn, frame_size)

            # chuyển frame_data về np.array
            frame = await decompress_frame(frame_data)

            # Đọc feature_data
            feature_data = await receive_all(client_conn, feature_data_size)

            # Giải mã feature_data
            feature = json.loads(feature_data.decode('utf-8'))

            keypoints_data = feature.get("keypoints_data", None)
            boxes_data = feature.get("boxes_data", None)
            body_color_data = feature.get("color_signature_data", None)
            uuid = feature.get("uuid", None)  # Lấy uuid từ feature

            # Tạo dictionary cho frame
            frame_dict = {
                "frame": frame
            }

            # Chỉ thêm vào frame_dict nếu các dữ liệu không phải là None
            if keypoints_data is not None:
                frame_dict["keypoints_data"] = keypoints_data
            if boxes_data is not None:
                frame_dict["boxes_data"] = boxes_data
            if body_color_data is not None:
                frame_dict["body_color_data"] = decode_color_data(body_color_data)
            if uuid is not None:
                frame_dict["uuid"] = uuid  # Thêm uuid vào frame_dict

            # Put vào frame_queue
            await put_frame_queue(frame_dict, frame_queue)

        except Exception as e:
            logger.error(f"Error worker {client_id}: {e}")
        finally:
            client_conn.close()

async def put_frame_queue(frame_data, frame_queue):
    try:
        if not frame_queue.full():
            frame_queue.put_nowait(frame_data)
            logger.info("Frame data added to frame_queue")
        else:
            logger.warning("Frame queue is full. Dropping frame.")
    except Exception as e:
        logger.error(f"Error in processing_frame: {e}")

async def receive_all(connection, size):
    loop = asyncio.get_event_loop()
    data = bytearray()
    while len(data) < size:
        chunk = await loop.sock_recv(connection, min(size - len(data), 4096))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data.extend(chunk)
    return bytes(data)

async def start_server(frame_queue, ramdisk, socket_path, server_index):
    """
    Khởi chạy server Unix socket bất đồng bộ và đo FPS.
    """
    ramdisk_path = create_ramdisk(ramdisk)  # Truyền tham số ramdisk vào hàm create_ramdisk
    if ramdisk_path is None:
        # Sử dụng RAMDisk (Ví dụ: tạo socket hoặc các thao tác khác)
        print("RAMDisk not created")
        exit(1)

    if not os.path.exists(os.path.dirname(socket_path)):
        os.makedirs(os.path.dirname(socket_path))
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(socket_path)
    server_socket.listen(128)
    server_socket.setblocking(False)
    logger.info(f"Server {server_index} listening on {socket_path}...")

    task_queue = asyncio.Queue()
    workers = [asyncio.create_task(worker(task_queue, frame_queue)) for _ in range(4)]

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            conn, _ = await asyncio.get_event_loop().sock_accept(server_socket)
            frame_count += 1

            # Gửi kết nối vào task queue
            await task_queue.put((conn, time.time()))

            # Tính toán FPS mỗi giây
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                logger.info(f"FPS_SERVER[{server_index}]: {fps:.2f}")  # Log FPS với server_index
                frame_count = 0
                start_time = current_time

    except asyncio.CancelledError:
        logger.info("Server task cancelled.")
    except Exception as e:
        logger.error(f"Error in server: {e}")
    finally:
        # Dừng tất cả worker tasks
        for _ in range(4):
            await task_queue.put((None, None))
        await asyncio.gather(*workers)

        server_socket.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        # Sau khi sử dụng xong, gỡ bỏ RAMDisk
        unmount_ramdisk()
        logger.info("Server stopped.")

async def main():
    # Khởi tạo frame_queue (queue này sẽ chứa các frame được gửi qua server)
    frame_queue = asyncio.Queue(maxsize=1000)

    # Khởi động server bất đồng bộ
    await start_server(frame_queue)
        

if __name__ == "__main__":
    asyncio.run(main())