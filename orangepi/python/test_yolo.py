from campc.orangepi.python.utils.yolo_pose_rknn2 import HumanDetection, core_mask
# from utils.logging_python_orangepi import setup_logging, get_logger
import cv2 
import os
# setup_logging()

if __name__ == '__main__':
    target = "rk3588"
    # core_mask = RKNN.NPU_CORE_0  # Sử dụng core 0 làm ví dụ 
    core_mask = core_mask[1]

    # Tạo instance của class
    detector = HumanDetection(target, core_mask)
    
    # Tải mô hình và khởi tạo runtime
    detector.load_model()
    detector.init_runtime()
     
    # # Đọc và xử lý hình ảnh
    # img = cv2.imread('file.jpg')
    # infer_img, aspect_ratio, offset_x, offset_y = detector.preprocess_image(img)
    # results = detector.inference(infer_img)
    # predbox = detector.postprocess(results, aspect_ratio, offset_x, offset_y)
    
    # # Vẽ kết quả và lưu hình ảnh
    # img_with_results = detector.draw_results(img, predbox)
    # cv2.imwrite("./result_rknn.jpg", img_with_results)
    # print("Saved image in ./result.jpg")
    
    # # Giải phóng tài nguyên
    # detector.release()
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("human_output", exist_ok=True)
    frame_k = 0

    source = cv2.VideoCapture("data/output_4k_video.mp4")
    while True:
        ret, frame = source.read()
        if not ret:
            print("Không thể đọc frame từ video")
            break

        predbox = detector.detect(frame)
        # Vẽ kết quả và lưu hình ảnh
        img_with_results = detector.draw_results(frame, predbox)
        frame_k +=1
        cv2.imwrite(f"human_output/result_rknn{frame_k}.jpg", img_with_results)
        print("Saved image in ./result.jpg")
        
    detector.release()
