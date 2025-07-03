import os
import socket
import threading
import queue as Queue  # Đổi tên module queue thành Queue để tránh xung đột
import time
import logging  # Import logging để ghi lại thông tin
import platform
import subprocess

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

SOCKET_PATH = "/mnt/ramdisk/ai_socket"  # Đường dẫn socket hợp lệ


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
                # Tạo RAMDisk mới
                subprocess.check_call(ramdisk_command, shell=True)

                # Định dạng và mount vào /Volumes/RAMDisk
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
            
            # Kiểm tra và tạo thư mục RAMDisk nếu chưa có
            if not os.path.exists(ramdisk_path):
                os.makedirs(ramdisk_path)

            try:
                # Tạo và mount RAMDisk vào thư mục /ramdisk với kích thước 16MB
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


def worker(task_queue):  # Đổi tên tham số queue thành task_queue
    while True:
        client_conn, client_id = task_queue.get()  # Sử dụng task_queue
        if client_conn is None:
            break  # Đến cuối công việc
        output_file = f"./output/python/frame_{client_id}.webp"  # Tạo file riêng cho mỗi client
        try:
            with open(output_file, "wb") as f:
                while True:
                    data = client_conn.recv(1024)
                    if not data:
                        break
                    f.write(data)
            # logging.info(f"Frame {client_id}")
        except Exception as e:
            logging.error(f"Error worker {client_id}: {e}")
        finally:
            client_conn.close()

def main():
    ramdisk_path = create_ramdisk()
    if ramdisk_path == None:
        # Sử dụng RAMDisk (Ví dụ: tạo socket hoặc các thao tác khác)
        print(f"RAMDisk not create")
        exit(1)

    SOCKET_PATH = ramdisk_path + "/ai_socket"

    logging.info(f"start {SOCKET_PATH}...")

    # Kiểm tra và tạo thư mục chứa socket nếu chưa có
    if not os.path.exists(os.path.dirname(SOCKET_PATH)):
        os.makedirs(os.path.dirname(SOCKET_PATH))

    # Xóa file socket nếu đã tồn tại
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    logging.info(f"start socket...")
    # Tạo socket Unix
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(5)
    logging.info(f"Listen {SOCKET_PATH}...")

    # Tạo Queue và Worker Threads
    task_queue = Queue.Queue()  # Đổi tên biến queue thành task_queue
    num_workers = 4  # Số lượng worker threads
    threads = []

    # Khởi tạo worker threads
    for _ in range(num_workers):
        thread = threading.Thread(target=worker, args=(task_queue,))
        thread.start()
        threads.append(thread)

    frame_number = 0
    start_time = time.time()  # Lưu thời gian bắt đầu
    sent_frames = 0  # Biến đếm số lượng frame đã nhận

    try:
        while True:
            conn, _ = server_socket.accept()
            frame_number += 1
            sent_frames += 1  # Tăng số lượng frame đã nhận

            # Đưa công việc vào queue cho worker
            task_queue.put((conn, frame_number))  # Sử dụng task_queue

            # Tính toán FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = sent_frames / elapsed_time  # Tính FPS
                start_time = current_time  # Cập nhật lại thời gian bắt đầu
                sent_frames = 0  # Reset số frame đã nhận
                logging.info(f"FPS: {fps:.2f}")  # In ra FPS

    except Exception as e:
        logging.error(f"Error server: {e}")
    finally:
        # Dừng tất cả worker threads
        for _ in range(num_workers):
            task_queue.put((None, None))  # Gửi tín hiệu kết thúc cho các worker
        for thread in threads:
            thread.join()

        server_socket.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        # Sau khi sử dụng xong, gỡ bỏ RAMDisk
        unmount_ramdisk()

        logging.info("Server stop.")

if __name__ == "__main__":
    main()
