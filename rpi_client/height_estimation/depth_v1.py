import numpy as np
import logging
from yolo_pose import HumanDetection

logger = logging.getLogger(__name__)

class DepthEstimation:
    def __init__(self, calibration_file='calibration_data.npz'):
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
        self.baseline = np.linalg.norm(self.T)  # Đơn vị phụ thuộc vào T (thường là mét)
        
        # Chuyển baseline từ mét sang cm nếu cần
        if self.baseline < 1:  # Giả định nếu nhỏ hơn 1 thì đơn vị là mét
            self.baseline *= 100  # Chuyển sang cm
            logger.info("Baseline được chuyển từ mét sang cm.")

        # Tính focal_length từ ma trận chiếu P1
        self.focal_length = self.P1[0, 0]  # Đơn vị: pixel
        self.human_detector = HumanDetection()  # Khởi tạo HumanDetection

        logger.info(f"Khởi tạo DepthEstimation với baseline={self.baseline:.2f} cm, focal_length={self.focal_length:.2f} pixel")

    def estimate_depth(self, left_image_path, right_image_path):
        """
        Tính độ sâu từ hai hình ảnh trái và phải.

        Args:
            left_image_path (str): Đường dẫn đến hình ảnh từ camera trái.
            right_image_path (str): Đường dẫn đến hình ảnh từ camera phải.

        Returns:
            list: Danh sách độ sâu (cm) cho từng người được phát hiện.
        """
        # Bước 1: Phát hiện keypoints từ hai hình ảnh
        keypoints_left, boxes_left = self.human_detector.run_detection(left_image_path)
        keypoints_right, boxes_right = self.human_detector.run_detection(right_image_path)

        # Kiểm tra nếu không có keypoints nào được phát hiện
        if keypoints_left.size == 0 or keypoints_right.size == 0:
            logger.warning("Không phát hiện được keypoints trong một hoặc cả hai hình ảnh.")
            return []

        # Bước 2: Kiểm tra số lượng người phát hiện được
        num_humans_left = len(keypoints_left)
        num_humans_right = len(keypoints_right)
        if num_humans_left != num_humans_right:
            logger.warning("Số người phát hiện được trong hai hình ảnh không khớp.")
            return []

        # Bước 3: Tính độ sâu cho từng người
        depths = []
        for i in range(num_humans_left):
            kp_left = keypoints_left[i]  # Keypoints của người thứ i trong ảnh trái
            kp_right = keypoints_right[i]  # Keypoints của người thứ i trong ảnh phải

            # Tính disparity trung bình từ tất cả các keypoints của người này
            disparities = []
            for j in range(len(kp_left)):
                x_L = kp_left[j][0]  # Tọa độ x trong ảnh trái
                x_R = kp_right[j][0]  # Tọa độ x trong ảnh phải
                disparity = x_R - x_L
                if disparity > 0:  # Chỉ thêm disparity hợp lệ (không âm)
                    disparities.append(disparity)

            if not disparities:
                logger.warning(f"Không có disparity hợp lệ cho người thứ {i+1}.")
                depths.append(None)
                continue

            # Tính disparity trung bình
            avg_disparity = np.mean(disparities)

            # Bước 4: Tính độ sâu
            depth = (self.baseline * self.focal_length) / avg_disparity
            depths.append(depth)
            logger.info(f"Độ sâu của người thứ {i+1}: {depth:.2f} cm")

        return depths

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO)

    # Khởi tạo DepthEstimation
    depth_estimator = DepthEstimation()

    # Đường dẫn đến hai hình ảnh
    left_image = "calibration3/cam1/frame_638.jpg"
    right_image = "calibration3/cam2/frame_640.jpg"

    # Tính độ sâu
    depths = depth_estimator.estimate_depth(left_image, right_image)
    for i, depth in enumerate(depths):
        if depth is not None:
            print(f"Người {i+1}: Độ sâu = {depth:.2f} cm")
        else:
            print(f"Người {i+1}: Không tính được độ sâu")