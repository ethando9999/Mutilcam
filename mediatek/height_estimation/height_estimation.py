import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression
import cv2 
import logging

logger = logging.getLogger(__name__)

class HeightEstimation:
    def __init__(self, dataset_path='dataset.csv', real_head_size_cm=20.0):
        """
        Khởi tạo lớp HeightEstimation.

        Args:
            dataset_path (str): Đường dẫn đến tập dữ liệu CSV.
            real_head_size_cm (float): Kích thước thực tế của đầu (cm), mặc định là 20 cm.
        """
        self.real_head_size_cm = real_head_size_cm
        self.dataset_path = dataset_path
        self.k = None
        self.offset = None
        self._build_conversion_function()

    def _build_conversion_function(self):
        """
        Xây dựng hàm chuyển đổi f(depth) = k * depth + offset từ tập dữ liệu.
        """
        # Đọc tập dữ liệu
        df = pd.read_csv(self.dataset_path)

        # Tính head_size_pixel dựa trên head_sizes_left và head_sizes_right (trung bình khi có đủ dữ liệu, tự bỏ qua NaN)
        df['head_size_pixel'] = df[['head_sizes_left', 'head_sizes_right']].mean(axis=1, skipna=True)
        
        # Tính scale = real_head_size_cm / head_size_pixel
        df['scale'] = self.real_head_size_cm / df['head_size_pixel']
        
        # Loại bỏ các dòng chứa NaN ở các cột quan trọng
        df = df.dropna(subset=['depth', 'head_size_pixel', 'scale'])
        
        if df.empty:
            raise ValueError("Không có dữ liệu hợp lệ sau khi loại bỏ các giá trị NaN. Vui lòng kiểm tra tập dữ liệu.")
        
        # Chuẩn bị dữ liệu cho hồi quy tuyến tính
        X = df['depth'].values.reshape(-1, 1)  # biến độc lập
        y = df['scale'].values  # biến phụ thuộc
        
        # Huấn luyện mô hình hồi quy tuyến tính
        model = LinearRegression()
        model.fit(X, y)
        
        # Lưu hệ số k và offset
        self.k = model.coef_[0]
        self.offset = model.intercept_
        
        logger.info(f"Hàm chuyển đổi: f(depth) = {self.k:.4f} * depth + {self.offset:.4f}")

    def estimate_height(self, depth, head_size_pixel):
        """
        Ước lượng chiều cao của con người từ độ sâu và kích thước đầu trong pixel.

        Args:
            depth (float): Độ sâu (khoảng cách từ camera đến người, cm).
            head_size_pixel (float): Kích thước đầu trong pixel.

        Returns:
            float: Chiều cao ước lượng của con người (cm).
        """
        if self.k is None or self.offset is None:
            raise ValueError("Hàm chuyển đổi chưa được xây dựng. Vui lòng kiểm tra tập dữ liệu.")

        # Tính scale từ depth bằng cách sử dụng hàm chuyển đổi
        scale = self.k * depth + self.offset
        
        # Tính kích thước đầu thực tế (cm)
        head_size_cm = scale * head_size_pixel
        
        # Ước tính chiều cao (ví dụ: nhân với 8)
        human_height = 8 * head_size_cm
        
        return human_height

def natural_keys(text):
    """
    Hàm sắp xếp theo thứ tự tự nhiên (natural sort)
    """
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

def visualize(image, depth, height):
    """
    Hiển thị thông tin độ sâu và chiều cao lên ảnh.

    Args:
        image (str hoặc np.ndarray): Đường dẫn tới ảnh hoặc mảng ảnh đã đọc.
        depth (float or None): Giá trị độ sâu (cm). Nếu không có giá trị hợp lệ, in ra 'N/A'.
        height (float or None): Giá trị chiều cao (cm). Nếu không có giá trị hợp lệ, in ra 'N/A'.
    """
    if isinstance(image, str) and os.path.isfile(image):
        img = cv2.imread(image)
    else:
        img = image.copy() if hasattr(image, 'copy') else image

    text_depth = f"Depth: {depth:.2f} cm" if depth is not None else "Depth: N/A"
    text_height = f"Height: {height:.2f} cm" if height is not None else "Height: N/A"

    pos_depth = (10, 30)
    pos_height = (10, 70)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color_depth = (0, 255, 0)
    color_height = (0, 0, 255)
    thickness = 2
    print(f"{img.shape[:2]}")

    cv2.putText(img, text_depth, pos_depth, font, font_scale, color_depth, thickness)
    cv2.putText(img, text_height, pos_height, font, font_scale, color_height, thickness)

    cv2.imshow("Visualization", img)
    cv2.waitKey(1)

def main():
    from yolo_pose import HumanDetection
    from depth_v1 import DepthEstimation
    from height_estimation import HeightEstimation

    folder_left = 'calibration6/cam1'
    folder_right = 'calibration6/cam2'

    images_left = sorted(glob.glob(os.path.join(folder_left, '*.jpg')), key=natural_keys)
    images_right = sorted(glob.glob(os.path.join(folder_right, '*.jpg')), key=natural_keys)

    depth_estimator = DepthEstimation()
    human_detector = HumanDetection()
    height_estimator = HeightEstimation()

    results = []

    for left_path, right_path in zip(images_left, images_right):
        keypoints_left_list, boxes_left_list = human_detector.run_detection(left_path)
        keypoints_right_list, boxes_right_list = human_detector.run_detection(right_path)

        num_humans_left = len(keypoints_left_list)
        num_humans_right = len(keypoints_right_list)

        if num_humans_left != num_humans_right:
            logger.warning("Số người phát hiện được trong hai hình ảnh không khớp: %s và %s", left_path, right_path)
            continue

        for i in range(num_humans_left):
            kp_left = keypoints_left_list[i]
            kp_right = keypoints_right_list[i]

            depth = depth_estimator.analyze(kp_left, kp_right)
            # Nếu depth không hợp lệ, bỏ qua mẫu này
            if depth is None:
                logger.warning("Không thể ước tính độ sâu cho cặp ảnh: %s và %s", left_path, right_path)
                continue

            head_size_left = human_detector.calculate_head_size(kp_left)
            head_size_right = human_detector.calculate_head_size(kp_right)

            if head_size_left is not None and head_size_right is not None:
                head_size_pixel = (head_size_left + head_size_right) / 2.0
            elif head_size_left is not None:
                head_size_pixel = head_size_left
            elif head_size_right is not None:
                head_size_pixel = head_size_right
            else:
                logger.warning("Không thể tính được kích thước đầu từ keypoints cho người thứ %d trong ảnh %s và %s",
                               i, left_path, right_path)
                continue

            estimated_height = height_estimator.estimate_height(depth, head_size_pixel)

            result = {
                "left_image": left_path,
                "right_image": right_path,
                "human_index": i,
                "depth": depth,
                "head_size_pixel": head_size_pixel,
                "estimated_height": estimated_height
            }
            results.append(result)
            print(f"Ảnh trái: {left_path}, ảnh phải: {right_path} - Người thứ {i+1}: Chiều cao ước tính = {estimated_height:.2f} cm")

            # Hiển thị kết quả trên ảnh trái (có thể chọn ảnh khác nếu mong muốn)
            visualize(left_path, depth, estimated_height)

    return results


# Ví dụ sử dụng
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    
    # # Khởi tạo HeightEstimation
    # height_estimator = HeightEstimation(dataset_path='dataset.csv', real_head_size_cm=20.0)
    
    # # Ví dụ với depth và head_size_pixel từ mẫu mới
    # sample_depth = 200.0  # cm
    # sample_head_size_pixel = 90.0  # pixel
    
    # # Ước tính chiều cao
    # estimated_height = height_estimator.estimate_height(sample_depth, sample_head_size_pixel)
    # print(f"Chiều cao ước lượng: {estimated_height:.2f} cm")
