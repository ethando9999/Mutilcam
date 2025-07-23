from track_local.track_wp import TrackingManager, start_track
from threading import Thread
import queue
import time

tracking_manager = TrackingManager(max_time_lost=15)
detection_queue = queue.Queue()

# 2. Khởi động luồng consumer (tracker)
# Chúng ta sử dụng daemon=True để luồng này tự động kết thúc khi chương trình chính thoát
tracker_thread = Thread(
    target=start_track,
    args=(detection_queue, tracking_manager),
    daemon=True
)
tracker_thread.start()

# 3. Vòng lặp producer giả lập (đây có thể là luồng xử lý video của bạn)
print("--- Starting detection producer simulation ---")

# Dữ liệu giả lập cho 2 đối tượng
mock_person_1 = {'bbox': [10, 10, 50, 100], 'world_point': [1.0, 2.0]}
mock_person_2 = {'bbox': [200, 50, 260, 180], 'world_point': [5.0, 3.5]}

# Mô phỏng trong 20 "khung hình"
for frame in range(20):
    print(f"\n>>> Producer: Simulating Frame {frame}")
    current_frame_detections = []

    # Di chuyển đối tượng 1
    mock_person_1['bbox'] = [x + 2 for x in mock_person_1['bbox']]
    mock_person_1['world_point'][0] += 0.1
    current_frame_detections.append(mock_person_1.copy())

    # Đối tượng 2 chỉ xuất hiện trong một số khung hình
    if 5 <= frame < 15:
        mock_person_2['bbox'] = [x - 1 for x in mock_person_2['bbox']]
        mock_person_2['world_point'][0] -= 0.05
        current_frame_detections.append(mock_person_2.copy())
        
    # Đối tượng 3 xuất hiện đột ngột
    if frame == 10:
            current_frame_detections.append({'bbox': [400, 100, 450, 250], 'world_point': [8.0, 1.0]})

    # Đưa dữ liệu của khung hình vào hàng đợi
    detection_queue.put(current_frame_detections)
    
    # Chờ một chút để mô phỏng thời gian thực
    time.sleep(0.5)

# 4. Gửi tín hiệu dừng cho luồng tracker
print("\n--- Producer finished. Sending stop signal to tracker thread. ---")
detection_queue.put(None)

# 5. Chờ luồng tracker kết thúc hoàn toàn
tracker_thread.join(timeout=5) # Chờ tối đa 5 giây
print("--- Main program finished. ---")

{
    "id": "3c9173df-ffe0-4d4a-ae97-273af8713022",
    "world_point": (92, 307)
}
