import cv2
import time
from yolo_pose import HumanDetection

# Khởi tạo camera (0 là camera mặc định)
source = cv2.VideoCapture("output_4k_video.mp4")   
detector = HumanDetection()

# Khởi tạo biến FPS
fps = 0
prev_time = time.time()

# Lấy kích thước khung hình video đầu vào
frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(source.get(cv2.CAP_PROP_FPS)) if source.get(cv2.CAP_PROP_FPS) > 0 else 30

# Khởi tạo VideoWriter để lưu video đầu ra
# output_video = cv2.VideoWriter(
#     'annotated_output.mp4',
#     cv2.VideoWriter_fourcc(*'mp4v'),  # Codec video
#     frame_fps,
#     (frame_width, frame_height)
# )

while True:
    ret, frame = source.read()
    if not ret:
        print("Không thể đọc frame từ video")
        break
    
    # Phát hiện người và keypoints 
    keypoints, boxes = detector.run_detection(frame)
    resolution = (frame_width, frame_height)
    # annotated_img = detector.draw_boxes_and_edges() 
    
    # # Ghi frame đã chú thích vào video
    # output_video.write(annotated_img)
    
    # Tính FPS 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Hiển thị FPS
    print(f'FPS NCNN: {fps:.2f}, size: {resolution[:2]}')
    
    # Thoát vòng lặp nếu nhấn phím 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Giải phóng tài nguyên
source.release()
# output_video.release()
cv2.destroyAllWindows()
