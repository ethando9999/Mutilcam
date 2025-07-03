from .send_frame_and_feature import start_sender
import asyncio

SERVER_IP = "192.168.1.123"
SERVER_PORT = [5050, 5051, 5052, 5053] 
server_address = [(SERVER_IP, port) for port in SERVER_PORT]  # Tạo danh sách server_address

async def start_send_manager(human_queue: asyncio.Queue, head_queue: asyncio.Queue, right_arm_queue: asyncio.Queue, left_arm_queue: asyncio.Queue):
    # Tạo danh sách các task sender
    tasks = [
        start_sender(human_queue, server_address[0]),  # Sử dụng server_address[0]
        start_sender(head_queue, server_address[1]),   # Sử dụng server_address[1]
        start_sender(right_arm_queue, server_address[2]),  # Sử dụng server_address[2]
        start_sender(left_arm_queue, server_address[3])   # Sử dụng server_address[3]
    ]
    
    # Chạy tất cả các task cùng lúc
    await asyncio.gather(*tasks)
