from utils.yolo_pose_rknn import HumanDetection, core_mask
# from utils.logging_python_orangepi import setup_logging, get_logger
import cv2 
import os
import time # Thêm thư viện time để có thể thêm độ trễ nếu cần

# setup_logging()

if __name__ == '__main__':
    target = "rk3588"
    core_mask = core_mask[1]

    # Tạo instance của class
    detector = HumanDetection(target, core_mask)
    
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

            # Thực hiện phát hiện trên mỗi frame
            predbox = detector.detect(frame)
            
            # Chỉ thực hiện vẽ và lưu nếu có đối tượng được phát hiện
            if predbox: # Điều kiện này đúng khi danh sách predbox không rỗng
                saved_frame_count += 1
                print(f"Phát hiện {len(predbox)} đối tượng! Đang lưu ảnh số {saved_frame_count}...")
                
                # Vẽ kết quả lên frame gốc
                img_with_results = detector.draw_results(frame, predbox)
                
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