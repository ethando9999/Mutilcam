from utils.yolo_pose import HumanDetection
# from utils.logging_python_orangepi import setup_logging, get_logger
import cv2 
import os
import time # Thêm thư viện time để có thể thêm độ trễ nếu cần

# setup_logging()

if __name__ == '__main__':

    # Tạo instance của class
    detector = HumanDetection()
    
    # Tạo thư mục nếu chưa tồn tại
    output_dir = "human_output"
    os.makedirs(output_dir, exist_ok=True)
    saved_frame_count = 0

    # Mở camera
    source = cv2.VideoCapture(0)
    if not source.isOpened():
        print("Lỗi: Không thể mở camera.")
        exit()

    print("Bắt đầu quá trình phát hiện... Nhấn Ctrl+C để dừng.")
    try:
        while True:
            ret, frame = source.read()
            if not ret:
                print("Không thể đọc frame từ video. Kết thúc.") 
                break
            start = time.time()
            # Thực hiện phát hiện trên mỗi frame
            keypoints_data, boxes_data = detector.run_detection(frame)
            fps = 1 / (time.time() - start)
            print(f"Features shape: {frame.shape}, FPS: {fps:.2f}")           
            
            # Chỉ thực hiện vẽ và lưu nếu có đối tượng được phát hiện
            if boxes_data: # Điều kiện này đúng khi danh sách predbox không rỗng
                saved_frame_count += 1
                print(f"Phát hiện {len(boxes_data)} đối tượng! Đang lưu ảnh số {saved_frame_count}...") 
                
                # Vẽ kết quả lên frame gốc
                img_with_results = detector.draw_boxes_and_edges()
                
                # Lưu hình ảnh có kết quả
                save_path = os.path.join(output_dir, f"result_rknn_{saved_frame_count}.jpg")
                cv2.imwrite(save_path, img_with_results)
                print(f"Đã lưu ảnh tại: {save_path}")
            
            # Bạn có thể thêm một khoảng nghỉ nhỏ ở đây nếu muốn giảm tải CPU
            # time.sleep(0.01)

    except KeyboardInterrupt:
        # Xử lý khi người dùng nhấn Ctrl+C
        print("\nĐã nhận tín hiệu dừng (Ctrl+C). Đang dọn dẹp và thoát.")
    finally:
        # Giải phóng tài nguyên
        source.release()
        # detector.rknn.release() # Gọi hàm giải phóng của RKNN nếu có
        print("Hoàn tất.")