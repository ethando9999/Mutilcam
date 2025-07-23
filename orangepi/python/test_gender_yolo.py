import time
import cv2
from ultralytics import YOLO

# 1. Load model (bạn cần cài ultralytics: pip install ultralytics)
model = YOLO("python/models/yolo11_gender_88test.pt")  # bạn có thể thay bằng yolov5n, m, l...

# 2. Mở ảnh (có thể là video, webcam hoặc thư mục)
img = cv2.imread("python/image2.jpg") 
if img is None:
    raise FileNotFoundError("Không tìm thấy file ảnh.")

# 3. Thông số
num_loops = 100  # số lần chạy inference để đo FPS chính xác hơn
total_time = 0.0

# 4. Vòng lặp inference
for i in range(num_loops):
    start = time.time()
    
    results = model(img)  # inference single image :contentReference[oaicite:4]{index=4}
    
    # Nếu bạn muốn vẽ bounding box để visualize
    annotated = results[0].plot()
    # cv2.imshow("YOLO", annotated)
    # cv2.waitKey(1)
    
    elapsed = time.time() - start
    total_time += elapsed

# cv2.destroyAllWindows()

# 5. Tính và in kết quả
avg_time = total_time / num_loops
fps = 1.0 / avg_time  # định nghĩa FPS = 1 / inference_time :contentReference[oaicite:5]{index=5}

print(f"Chạy {num_loops} lần → thời gian trung bình: {avg_time:.4f}s, FPS ≈ {fps:.2f}")
