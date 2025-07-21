import asyncio
import datetime
import uuid
import random  # Thêm thư viện random

# Đảm bảo rằng file này nằm ở thư mục gốc để có thể import từ thư mục 'core'
from core.socket_sender import start_socket_sender

async def produce_data(socket_queue: asyncio.Queue):
    """
    Hàm này sẽ tạo ra một gói dữ liệu mẫu và đưa vào hàng đợi mỗi giây.
    Tọa độ point3D sẽ được thay đổi theo yêu cầu:
    - Phần tử 1: Tăng đều từ 10 đến 200 rồi lặp lại.
    - Phần tử 2: Dao động ngẫu nhiên từ 290 đến 320.
    - Phần tử 3: Luôn bằng 0.
    """
    frame_counter = 0
    # Khởi tạo giá trị ban đầu cho tọa độ x
    x_coord = 10
    pid = str(uuid.uuid4())
    while True:
        frame_counter += 1
        
        # Cập nhật tọa độ x, nếu vượt 200 thì reset về 10
        if x_coord > 200:
            x_coord = 10
            
        # Dữ liệu mẫu dựa trên hình ảnh được cung cấp
        packet = {
          "frame_id": frame_counter,
          "person_id": pid,
          "gender": "Male",
          "race": "None",
          "age": "Adult",
          "height": 0.0,
          "time_detect": datetime.datetime.now().isoformat(),
          "camera_id": "cam_01",
          "point3D": [
            x_coord,                # Phần tử 1: Tăng đều
            random.randint(290, 320), # Phần tử 2: Dao động ngẫu nhiên
            0                       # Phần tử 3: Hằng số 0
          ]
        }
        
        try:
            # Thử đưa gói tin mới vào hàng đợi mà không chờ đợi
            socket_queue.put_nowait(packet)
            print(f"-> Đã đưa vào hàng đợi frame_id: {frame_counter}, size: {socket_queue.qsize()}/{socket_queue.maxsize}")
        except asyncio.QueueFull:
            # Nếu hàng đợi đầy, lấy gói tin cũ nhất ra
            old_packet = socket_queue.get_nowait()
            print(f"⚠️  Hàng đợi đầy. Đã loại bỏ frame_id cũ: {old_packet['frame_id']}")
            
            # Sau đó đưa gói tin mới vào (bây giờ chắc chắn sẽ có chỗ)
            socket_queue.put_nowait(packet)
            print(f"-> Đã đưa vào hàng đợi frame_id: {frame_counter} (thay thế), size: {socket_queue.qsize()}/{socket_queue.maxsize}")
        # ------------------------------------------

        x_coord += 1
        await asyncio.sleep(1)

async def main():
    """
    Hàm chính để thiết lập hàng đợi và khởi chạy các tác vụ producer và consumer.
    """
    # URI của WebSocket server.
    # Cần phải có một server đang chạy ở địa chỉ này để test.
    server_uri = "ws://192.168.1.67:9090/api/ws/camera"
    socket_queue = asyncio.Queue(maxsize=1)

    print("Khởi tạo Socket Sender và Producer...")

    # Khởi chạy tác vụ tạo dữ liệu (producer)
    producer_task = asyncio.create_task(produce_data(socket_queue))
    sender_task = asyncio.create_task(start_socket_sender(socket_queue, server_uri))
    # Chạy cả hai tác vụ đồng thời
    await asyncio.gather(sender_task, producer_task)

if __name__ == "__main__":
    print("Bắt đầu kịch bản kiểm thử...")
    print("Đảm bảo rằng bạn đã khởi chạy một WebSocket server trên ws://localhost:8765")
    print("Bạn có thể chạy lệnh: python -m websockets ws://localhost:8765")
    print("Nhấn CTRL+C để dừng.")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nĐã dừng kịch bản.")