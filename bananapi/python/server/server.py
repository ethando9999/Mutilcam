import os
import socket
import threading
import asyncio
import time
import logging
import platform
import subprocess

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

SOCKET_PATH = "/mnt/ramdisk/ai_socket"  # Giữ nguyên đường dẫn socket


def check_existing_ramdisk():
    """Kiểm tra xem đã có RAMDisk nào được mount chưa."""
    system_platform = platform.system()

    if system_platform == "Darwin":  # macOS 
        try:
            # Lệnh để lấy tất cả các ổ đĩa đã được mount
            mount_output = subprocess.check_output("mount", shell=True).decode("utf-8")
            # Kiểm tra xem có thư mục nào chứa "RAMDisk" không
            for line in mount_output.splitlines():
                if "RAMDisk" in line and "/Volumes/RAMDisk" in line:
                    print("RAMDisk already mounted at /Volumes/RAMDisk")
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
                if "tmpfs" in line and "ramdisk" in line:
                    print("RAMDisk already mounted at /ramdisk")
                    return True  # Đã có RAMDisk, không cần tạo mới
            return False  # Không có RAMDisk nào, cần tạo mới
        except subprocess.CalledProcessError as e:
            print(f"Error checking mount: {e}")
            return False
    else:
        print("This script only works on macOS and Linux.")
        return False

def create_ramdisk(): 
    """Tạo RAMDisk nếu chưa có."""
    system_platform = platform.system()

    if system_platform == "Darwin":  # macOS
        if not check_existing_ramdisk():
            print("Creating new RAMDisk on macOS...")
            ramdisk_size_in_blocks = 32768  # 16MB ÷ 512 bytes = 32768 blocks
            ramdisk_command = f"diskutil erasevolume HFS+ RAMDisk `hdiutil attach -nomount ram://{ramdisk_size_in_blocks}`"
            
            try:
                subprocess.check_call(ramdisk_command, shell=True)
                print("RAMDisk created and mounted at /Volumes/RAMDisk.")
                return "/Volumes/RAMDisk"
            except subprocess.CalledProcessError as e:
                print(f"Error creating RAMDisk: {e}")
                return None
        else:
            return "/Volumes/RAMDisk"

    elif system_platform == "Linux" or system_platform == "Linux2":  # Linux / Raspberry Pi
        if not check_existing_ramdisk():
            print("Creating new RAMDisk on Linux/Raspberry Pi...")
            ramdisk_path = "/mnt/ramdisk"
            
            try:
                # Sử dụng sudo để tạo thư mục RAMDisk nếu chưa có
                subprocess.check_call(f"sudo mkdir -p {ramdisk_path}", shell=True)
                
                # Mount RAMDisk với kích thước 16MB
                subprocess.check_call(f"sudo mount -t tmpfs -o size=16M tmpfs {ramdisk_path}", shell=True)
                print(f"RAMDisk created and mounted at {ramdisk_path}.")
                return ramdisk_path
            except subprocess.CalledProcessError as e:
                print(f"Error creating RAMDisk: {e}")
                return None
        else:
            return "/mnt/ramdisk"

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

async def worker(task_queue, frame_queue):
    """Worker xử lý kết nối."""
    while True:
        client_conn, client_id = await task_queue.get()
        if client_conn is None:
            break  # Kết thúc khi nhận tín hiệu dừng

        try:
            loop = asyncio.get_event_loop()
            frame_data = bytearray()
            while True:
                data = await loop.sock_recv(client_conn, 1024)
                if not data:
                    break
                frame_data.extend(data)

            if frame_data and not frame_queue.full():
                await frame_queue.put(bytes(frame_data))
                logging.info("Frame data added to frame_queue")
            else:
                logging.warning("Frame queue is full. Dropping frame.") 

        except Exception as e:
            logging.error(f"Error worker {client_id}: {e}")
        finally:
            client_conn.close()

async def start_server(frame_queue):
    """
    Khởi chạy server Unix socket bất đồng bộ và đo FPS.
    """
    ramdisk_path = create_ramdisk()
    if ramdisk_path == None:
        # Sử dụng RAMDisk (Ví dụ: tạo socket hoặc các thao tác khác)
        print(f"RAMDisk not create")
        exit(1)
    if not os.path.exists(os.path.dirname(SOCKET_PATH)):
        os.makedirs(os.path.dirname(SOCKET_PATH))
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    # Tăng buffer size cho socket 
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 300 * 1024 * 1024)  # 4MB receive buffer 
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 300 * 1024 * 1024)  # 4MB send buffer 
    
    # Tăng backlog queue
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(1000)  # Tăng từ 5 lên 128
    server_socket.setblocking(False)
    
    logging.info(f"Listening on {SOCKET_PATH} with increased buffer size and backlog...")

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
                logging.info(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = current_time

    except asyncio.CancelledError:
        logging.info("Server task cancelled.")
    except Exception as e:
        logging.error(f"Error in server: {e}")
    finally:
        # Dừng tất cả worker tasks
        for _ in range(4):
            await task_queue.put((None, None))
        await asyncio.gather(*workers)

        server_socket.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        logging.info("Server stopped.")

async def main():
    # Khởi tạo frame_queue (queue này sẽ chứa các frame được gửi qua server) 
    frame_queue = asyncio.Queue(maxsize=1000)

    # Khởi động server bất đồng bộ
    await start_server(frame_queue)

if __name__ == "__main__":
    asyncio.run(main())
