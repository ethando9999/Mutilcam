import cv2
import numpy as np
import logging
from yolo_pose import HumanDetection

logger = logging.getLogger(__name__)

class DepthEstimation:
    def __init__(self, calibration_file='calibration_data1.npz', real_baseline=5.5):
        """
        Khởi tạo lớp DepthEstimation với các tham số từ tệp hiệu chuẩn.

        Args:
            calibration_file (str): Đường dẫn đến tệp .npz chứa tham số hiệu chuẩn.
        """
        # Tải tham số hiệu chuẩn
        calib_data = np.load(calibration_file)
        self.mtx_l = calib_data['mtx_l']
        self.dist_l = calib_data['dist_l']
        self.mtx_r = calib_data['mtx_r']
        self.dist_r = calib_data['dist_r']
        self.R1 = calib_data['R1']
        self.R2 = calib_data['R2']
        self.P1 = calib_data['P1']
        self.P2 = calib_data['P2']
        self.Q = calib_data['Q']
        self.T = calib_data['T']
        
        # Tính baseline từ vector tịnh tiến T
        self.baseline_calib = np.linalg.norm(self.T)  # Đơn vị phụ thuộc vào T (thường là mét)
        self.baseline = real_baseline
        self.scale = self.baseline / self.baseline_calib
        
        # Chuyển baseline từ mét sang cm nếu cần
        if self.baseline < 1:  # Giả định nếu nhỏ hơn 1 thì đơn vị là mét
            self.baseline *= 100  # Chuyển sang cm
            logger.info("Baseline được chuyển từ mét sang cm.")
        print("baseline: ", self.baseline)

        self.focal_length = self.P1[0, 0] 
        print("focal_length: ", self.focal_length)
        # self.human_detector = HumanDetection()  # Khởi tạo HumanDetection

        logger.info(f"Khởi tạo DepthEstimation với baseline={self.baseline:.2f} cm, focal_length={self.focal_length:.2f} pixel")

    def compute_depth_from_keypoints(self, head_kp_left, head_kp_right):
        disparities = []
        for kp_l, kp_r in zip(head_kp_left, head_kp_right):
            x_L = kp_l[0]
            x_R = kp_r[0]
            disparity = x_R - x_L
            if disparity > 0:
                disparities.append(disparity)

        if not disparities:
            logger.warning("Không có disparity hợp lệ.")
            return None

        avg_disparity = np.mean(disparities)
        depth = (self.baseline * self.focal_length) / avg_disparity
        return depth    

    def analyze(self, keypoints_left, keypoints_right):
        """
        Ước tính độ sâu cho một người duy nhất từ hai hình ảnh trái và phải, sử dụng keypoints vùng đầu.
        Chỉ tính độ sâu nếu toàn bộ keypoints vùng đầu đều hợp lệ (khác [0, 0]).

        Args:
            keypoints_left (list): Danh sách keypoints từ ảnh trái của một người.
            keypoints_right (list): Danh sách keypoints từ ảnh phải của một người.

        Returns:
            float or None: Độ sâu (cm) nếu tính được, ngược lại trả về None.
        """
        # Kiểm tra dữ liệu đầu vào
        if keypoints_left is None or keypoints_right is None:
            logger.warning("Thiếu keypoints trong một trong hai ảnh.")
            return None

        if len(keypoints_left) == 0 or len(keypoints_right) == 0:
            logger.warning("Không có keypoints trong ảnh trái hoặc phải.")
            return None

        # Chỉ số keypoints vùng đầu theo COCO: Nose, left eye, right eye, left ear, right ear
        head_keypoints_indices = [0, 1, 2, 3, 4]

        # Lấy keypoints vùng đầu đầy đủ cho cả hai ảnh
        try:
            head_kp_left = [keypoints_left[j] for j in head_keypoints_indices]
            head_kp_right = [keypoints_right[j] for j in head_keypoints_indices]
        except IndexError:
            logger.warning("Không đủ keypoints vùng đầu.")
            return None

        # Kiểm tra tất cả keypoints vùng đầu phải hợp lệ (khác [0, 0])
        if any((kp[0] == 0 and kp[1] == 0) for kp in head_kp_left + head_kp_right):
            logger.warning("Keypoints vùng đầu không đầy đủ hoặc không hợp lệ.")
            return None

        # Tính disparity từ các keypoints vùng đầu
        disparities = []
        for kp_l, kp_r in zip(head_kp_left, head_kp_right):
            x_L = kp_l[0]
            x_R = kp_r[0]
            disparity = x_R - x_L
            if disparity > 0:
                disparities.append(disparity)

        if not disparities:
            logger.warning("Không có disparity hợp lệ.")
            return None

        # Tính độ sâu
        avg_disparity = np.mean(disparities)
        depth = (self.baseline * self.focal_length) / avg_disparity
        return depth


def natural_keys(text):
    """
    Hàm sắp xếp theo thứ tự tự nhiên (natural sort)
    """
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

def evaluate_depth(folder_left, folder_right):
    """
    Đánh giá độ sâu từ các thư mục chứa ảnh của camera trái và phải.

    Args:
        folder_left (str): Đường dẫn đến thư mục ảnh camera trái.
        folder_right (str): Đường dẫn đến thư mục ảnh camera phải.
    """
    import glob
    import os
    # Đọc danh sách ảnh của camera trái và phải và sắp xếp theo thứ tự tự nhiên
    images_left = sorted(glob.glob(os.path.join(folder_left, '*.jpg')), key=natural_keys)
    images_right = sorted(glob.glob(os.path.join(folder_right, '*.jpg')), key=natural_keys)

    # Khởi tạo DepthEstimation
    depth_estimator = DepthEstimation()

    for left_path, right_path in zip(images_left, images_right):
        print(f"left_path: {left_path}")
        keypoints_left, keypoints_right, head_sizes_left, head_sizes_right = depth_estimator.detect_human(left_path, right_path)

        # Tính độ sâu
        depths = depth_estimator.estimate_depth(keypoints_left, keypoints_right, head_sizes_left, head_sizes_right)
        for i, depth in enumerate(depths):
            if depth is not None:
                print(f"Người {i+1}: Độ sâu = {depth:.2f} cm, Kích thước đầu trái = {head_sizes_left[i] if head_sizes_left[i] is not None else 'N/A'} pixel, Kích thước đầu phải = {head_sizes_right[i] if head_sizes_right[i] is not None else 'N/A'} pixel")
            else:
                print(f"Người {i+1}: Không tính được độ sâu, Kích thước đầu trái = {head_sizes_left[i] if head_sizes_left[i] is not None else 'N/A'} pixel, Kích thước đầu phải = {head_sizes_right[i] if head_sizes_right[i] is not None else 'N/A'} pixel")

def visualize(image, depth):
    """
    Hiển thị thông tin độ sâu lên ảnh.

    Args:
        image (str hoặc np.ndarray): Đường dẫn tới ảnh hoặc mảng ảnh đã đọc.
        depth (float or None): Giá trị độ sâu (cm). Nếu không có giá trị hợp lệ, in ra 'N/A'.
    """
    import os
    if isinstance(image, str) and os.path.isfile(image):
        img = cv2.imread(image)
    else:
        img = image.copy() if hasattr(image, 'copy') else image

    text_depth = f"Depth: {depth:.2f} cm" if depth is not None else "Depth: N/A"

    pos_depth = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color_depth = (0, 255, 0)
    thickness = 2

    cv2.putText(img, text_depth, pos_depth, font, font_scale, color_depth, thickness)

    cv2.imshow("Visualization", img)
    cv2.waitKey(1)


def create_dataset(folder_left, folder_right, real_head_size_cm=20.0):
    import glob
    import os
    import csv

    images_left = sorted(glob.glob(os.path.join(folder_left, '*.jpg')), key=natural_keys)
    images_right = sorted(glob.glob(os.path.join(folder_right, '*.jpg')), key=natural_keys)

    depth_estimator = DepthEstimation()
    human_detector = HumanDetection()

    with open('dataset1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['left_path', 'right_path', 'depth', 'head_sizes_left', 'head_sizes_right', 'real_head_size_cm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for left_path, right_path in zip(images_left, images_right):
            keypoints_left, boxes_left_list = human_detector.run_detection(left_path)
            keypoints_right, boxes_right_list = human_detector.run_detection(right_path)

            head_sizes_left = [human_detector.calculate_head_size(kp) for kp in keypoints_left] if keypoints_left.size > 0 else []
            head_sizes_right = [human_detector.calculate_head_size(kp) for kp in keypoints_right] if keypoints_right.size > 0 else []

            print(f"left_path: {left_path}")
            num_persons = min(len(keypoints_left), len(keypoints_right))

            for i in range(num_persons):
                head_kp_left = human_detector.extract_head_keypoints(keypoints_left[i])
                head_kp_right = human_detector.extract_head_keypoints(keypoints_right[i])
                depth = depth_estimator.compute_depth_from_keypoints(head_kp_left, head_kp_right)
                if head_sizes_left[i] is None and head_sizes_right[i] is None:
                    continue
                if depth is not None:
                    print(f"Người {i+1}: Độ sâu = {depth:.2f} cm, Kích thước đầu trái = {head_sizes_left[i]} pixel, Kích thước đầu phải = {head_sizes_right[i]}")
                visualize(left_path, depth)
                writer.writerow({
                    'left_path': left_path,
                    'right_path': right_path,
                    'depth': f"{depth:.2f}" if depth is not None else "",
                    'head_sizes_left': head_sizes_left[i],
                    'head_sizes_right': head_sizes_right[i],
                    'real_head_size_cm': real_head_size_cm,
                })

        

def main():
    # Khởi tạo DepthEstimation
    depth_estimator = DepthEstimation()

    # Đường dẫn đến hai hình ảnh
    left_image = "calibration4/cam1/frame_35.jpg"
    right_image = "calibration4/cam2/frame_35.jpg"

    # Phát hiện keypoints và tính kích thước đầu
    keypoints_left, keypoints_right, head_sizes_left, head_sizes_right = depth_estimator.detect_human(left_image, right_image)
    depths = depth_estimator.estimate_depth(keypoints_left, keypoints_right, head_sizes_left, head_sizes_right)
    for i, depth in enumerate(depths):
        if depth is not None:
            print(f"Người {i+1}: Độ sâu = {depth:.2f} cm")
        else:
            print(f"Người {i+1}: Không tính được độ sâu")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO)
    folder_cam_left = 'dataset1/cam1'
    folder_cam_right = 'dataset1/cam2'
    # evaluate_depth(folder_cam_left, folder_cam_right)
    create_dataset(folder_cam_left, folder_cam_right, real_head_size_cm=22)