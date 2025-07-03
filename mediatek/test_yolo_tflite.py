import torch
from PIL import Image
import numpy as np

# Import class TFLiteBackend từ file trước đó (giả sử cùng thư mục)
from yolo_tfite import TFLiteBackend  # Thay bằng tên file chứa class TFLiteBackend nếu khác

def preprocess_image(image_path, img_size=(640, 640)):
    """
    Tiền xử lý ảnh đầu vào: load, resize, và chuyển thành tensor.

    Args:
        image_path (str): Đường dẫn đến file ảnh.
        img_size (tuple): Kích thước đầu vào của model (height, width).

    Returns:
        torch.Tensor: Tensor ảnh với shape (1, 3, height, width).
    """
    # Load ảnh
    img = Image.open(image_path).convert('RGB')
    
    # Resize ảnh về kích thước mong muốn
    img = img.resize(img_size, Image.Resampling.LANCZOS)
    
    # Chuyển thành numpy array và normalize về [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0  
    
    # Chuyển từ (H, W, C) sang (C, H, W)
    img_np = img_np.transpose(2, 0, 1)
    
    # Thêm batch dimension và chuyển thành torch tensor
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    
    return img_tensor

def main():
    backend = TFLiteBackend()

    image_path = "frame_66.jpg"
    input_tensor = preprocess_image(image_path=image_path)

    # Chạy inference
    print("Running inference...")
    output = backend.forward(input_tensor)

    # In thông tin output
    print("\nInference completed. Output details:")
    if isinstance(output, list):
        for i, out in enumerate(output):
            print(f"Output {i}: shape = {out.shape}, dtype = {out.dtype}")
    else:
        print(f"Output: shape = {output.shape}, dtype = {output.dtype}")

if __name__ == "__main__":
    main()