import asyncio
from utils.logging_python_orangepi import get_logger
from .server import start_server

logger = get_logger(__name__)

async def run_servers(human_queue: asyncio.Queue, head_queue: asyncio.Queue, right_arm_queue: asyncio.Queue, left_arm_queue: asyncio.Queue):
    """Khởi chạy nhiều server với các thông số khác nhau."""
    ramdisks = ["/mnt/ramdisk1", "/mnt/ramdisk2", "/mnt/ramdisk3", "/mnt/ramdisk4"]  # Danh sách đường dẫn RAMDisk
    # Bỏ ai_sockets
    # num_socket = len(ramdisks)  # Số lượng socket
    # ai_sockets = [f"ai_socket{i+1}" for i in range(num_socket)]  # Tạo danh sách ai_socket dựa trên num_socket
    
    # Tạo danh sách socket_path từ ramdisks
    socket_paths = [f"{ramdisk}/socket" for ramdisk in ramdisks]  # Thay đổi để chỉ sử dụng ramdisks 

    # Danh sách các hàng đợi để truyền vào start_server
    queues = [human_queue, head_queue, right_arm_queue, left_arm_queue]

    # Khởi động server cho từng socket
    tasks = []
    for index, socket_path in enumerate(socket_paths):
        # Lấy hàng đợi tương ứng với index, nếu index lớn hơn số lượng hàng đợi thì sử dụng hàng đợi cuối cùng
        queue_to_use = queues[index] if index < len(queues) else queues[-1]
        tasks.append(asyncio.create_task(start_server(queue_to_use, ramdisks[index % len(ramdisks)], socket_path, index)))

    await asyncio.gather(*tasks)  # Chờ tất cả các server hoàn thành