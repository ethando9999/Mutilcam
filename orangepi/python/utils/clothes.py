import cv2
import numpy as np
from rknn.api import RKNN

# Định nghĩa các lớp quần áo
NUM_CLASSES = 15
class_names = [
    "Blazer", "Coat", "Denim Jacket", "Dresses", "Hoodie",
    "Jacket", "Jeans", "Long Pants", "Polo", "Shirt",
    "Shorts", "Skirt", "Sports Jacket", "Sweater", "T-shirt"
]

def preprocess_image(image_path, input_size=(224, 224)):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image {image_path}")

    # Resize về kích thước đầu vào
    img = cv2.resize(img, input_size)
    
    # Chuyển đổi sang RGB và chuẩn hóa
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Chuyển sang định dạng NHWC
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # Thêm batch dimension
    return img

def main():
    # Đường dẫn đến mô hình và ảnh
    model_path = "/home/ubuntu/orangepi/rknn/clothes_image_detection.rknn"  # Đường dẫn đã sửa
    image_path = "/home/ubuntu/orangepi/image.jpg"

    # Khởi tạo RKNN 
    rknn = RKNN() 
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("Error: Failed to load RKNN model")
        exit(ret)

    print("Initializing runtime environment...") 
    ret = rknn.init_runtime()
    if ret != 0:
        print("Error: Failed to initialize runtime")
        rknn.release()
        exit(ret)

    # Tiền xử lý ảnh
    input_data = preprocess_image(image_path)

    # Chạy suy luận
    print("Running inference...")
    outputs = rknn.inference(inputs=[input_data])
    
    # Xử lý đầu ra
    output_data = outputs[0].reshape(-1)  # Giả định đầu ra là softmax
    max_index = np.argmax(output_data)
    max_prob = output_data[max_index]

    # In kết quả
    print(f"Predicted class: {class_names[max_index]} (Probability: {max_prob*100:.2f}%)")

    # Giải phóng tài nguyên
    rknn.release()

if __name__ == "__main__":
    main()