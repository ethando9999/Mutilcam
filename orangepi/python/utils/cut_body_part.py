import cv2
import numpy as np
from .logging_python_orangepi import get_logger

logger = get_logger(__name__)

COCO_KEYPOINTS = [
    (0, "Mũi"),          # 0: Mũi
    (1, "Mắt trái"),     # 1: Mắt trái
    (2, "Mắt phải"),     # 2: Mắt phải
    (3, "Tai trái"),     # 3: Tai trái
    (4, "Tai phải"),     # 4: Tai phải
    (5, "Vai trái"),     # 5: Vai trái
    (6, "Vai phải"),     # 6: Vai phải
    (7, "Khuỷu tay trái"), # 7: Khuỷu tay trái
    (8, "Khuỷu tay phải"), # 8: Khuỷu tay phải
    (9, "Cổ tay trái"),  # 9: Cổ tay trái
    (10, "Cổ tay phải"), # 10: Cổ tay phải
    (11, "Hông trái"),    # 11: Hông trái
    (12, "Hông phải"),    # 12: Hông phải
    (13, "Đầu gối trái"),  # 13: Đầu gối trái
    (14, "Đầu gối phải"),  # 14: Đầu gối phải
    (15, "Mắt cá chân trái"), # 15: Mắt cá chân trái
    (16, "Mắt cá chân phải")  # 16: Mắt cá chân phải
]

def compute_head_bbox(person_keypoints, scale=1, adjust_ratio=0.35):
    """
    Tính toán vùng giới hạn (bounding box) cho vùng đầu dựa trên keypoints.
    Sử dụng tâm và bán kính lớn nhất để tạo vùng cắt hình vuông.
    
    Args:
        person_keypoints (list of tuples): Danh sách tọa độ các keypoints [(x, y), ...].
        scale (float): Hệ số mở rộng vùng cắt.
        adjust_ratio (float): Tỷ lệ điều chỉnh center_y hướng lên trên (Được sinh ra do phần trọng tâm bị lệnh xuống dưới khi có thêm vai trái và vai phải)
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) nếu hợp lệ, None nếu không có keypoints.
    """
    # Lọc bỏ các keypoints có tọa độ (0, 0)
    keypoints = np.array([kp for kp in person_keypoints if not (kp[0] == 0 and kp[1] == 0)])

    if keypoints.shape[0] == 0:  # Không có keypoint hợp lệ
        return None

    # Tính tâm trung bình
    center_x = np.mean(keypoints[:, 0])
    center_y = np.mean(keypoints[:, 1])

    # Tính bán kính lớn nhất
    distances = np.linalg.norm(keypoints - [center_x, center_y], axis=1)
    radius = np.max(distances) * scale

    # Điều chỉnh center_y hướng lên trên
    center_y -= center_y * adjust_ratio  

    # Xác định tọa độ vùng cắt hình vuông
    x_min = int(center_x - radius)
    x_max = int(center_x + radius)
    y_min = int(center_y - radius)
    y_max = int(center_y + radius)

    return x_min, y_min, x_max, y_max

def extract_head_from_frame(frame, person_keypoints, scale=1):
    """
    Tính toán bounding box vùng đầu từ frame dựa trên keypoints.

    Args:
        frame (numpy array): Ảnh đầu vào.
        person_keypoints (list of tuples): Keypoints của khuôn mặt [(x, y), ...].
        scale (float): Hệ số mở rộng vùng cắt.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) hoặc None nếu không hợp lệ.
    """
    try:
        keypoints = [
            person_keypoints[0],  # Mũi
            person_keypoints[1],  # Mắt trái
            person_keypoints[2],  # Mắt phải
            person_keypoints[3],  # Tai trái
            person_keypoints[4],  # Tai phải
            person_keypoints[5],  # Vai trái
            person_keypoints[6]   # Vai phải
        ]
        bbox = compute_head_bbox(keypoints, scale)
        if bbox is None:
            logger.error("No valid keypoints available for head extraction.")
            return None

        x_min, y_min, x_max, y_max = bbox

        # Giới hạn trong kích thước frame
        x_min = max(0, x_min)
        x_max = min(frame.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(frame.shape[0], y_max)

        if x_max <= x_min or y_max <= y_min:
            logger.error("Invalid bounding box for head region.")
            return None

        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        logger.error(f"Error in extract_head_from_frame: {e}, person_keypoints: {person_keypoints}")
        return None


def compute_arm_region(frame, start_point, end_point, thickness=100):
    """
    Tính toán mask vùng cánh tay dựa trên đường thẳng giữa start_point và end_point với độ dày được mở rộng theo tỉ lệ.
    """
    # Chuyển đổi start_point và end_point sang kiểu int
    start_point_int = tuple(map(int, start_point))
    end_point_int = tuple(map(int, end_point))
    
    # Tính độ dài của đường thẳng
    line_length = np.linalg.norm(np.array(start_point) - np.array(end_point))
    
    # Tính new_thickness theo tỷ lệ giữa chiều cao frame và độ dài đường thẳng
    frame_height = frame.shape[0]
    new_thickness = int(thickness * (frame_height / line_length))
    
    # Tạo một mask để vẽ đường thẳng
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Vẽ đường thẳng trên mask
    cv2.line(mask, start_point_int, end_point_int, 255, new_thickness)
    
    return mask

def extract_arm_region(frame, start_point, end_point, thickness=25):
    """
    Tính toán bounding box vùng cánh tay dựa vào 2 điểm và độ dày.

    Args:
        frame (numpy array): Ảnh đầu vào.
        start_point (tuple): Điểm bắt đầu (x, y).
        end_point (tuple): Điểm kết thúc (x, y).
        thickness (int): Độ dày ước lượng vùng cánh tay.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) hoặc None nếu không hợp lệ.
    """
    try:
        if (start_point[0] == 0 and start_point[1] == 0) or (end_point[0] == 0 and end_point[1] == 0):
            return None

        mask = compute_arm_region(frame, start_point, end_point, thickness)
        x, y, w, h = cv2.boundingRect(mask)

        if w == 0 or h == 0:
            logger.error("Invalid bounding box for arm region.")
            return None

        return (x, y, x + w, y + h)
    except Exception as e:
        logger.error(f"Error in extract_arm_region: {e}, start_point: {start_point}, end_point: {end_point}")
        return None

def extract_arm_region2(frame, start_point, end_point, scale=4):
    """
    Cắt vùng cánh tay từ frame theo đường thẳng giữa start_point và end_point với độ dày nhất định.
    """
    try:
        # Kiểm tra và loại bỏ các điểm có tọa độ [0, 0]
        if (start_point[0] == 0 and start_point[1] == 0) or (end_point[0] == 0 and end_point[1] == 0):
            logger.error(f"Invalid keypoints: start_point: {start_point}, end_point: {end_point} - One or both points are [0, 0].")
            return None  # Trả về None nếu có điểm không hợp lệ

        # Chuyển đổi tọa độ thành số nguyên
        start_x, start_y = int(start_point[0]), int(start_point[1])
        end_x, end_y = int(end_point[0]), int(end_point[1])

        # Tính toán chiều rộng và chiều cao của vùng cắt
        width = abs(end_x - start_x)
        height = abs(end_y - start_y)

        # Mở rộng vùng cắt theo tỉ lệ
        start_x = int(start_x - (width * (scale - 1) / 2))
        end_x = int(end_x + (width * (scale - 1) / 2))
        start_y = int(start_y - (height * (scale - 1) / 2))
        end_y = int(end_y + (height * (scale - 1) / 2))

        # Đảm bảo tọa độ không vượt quá kích thước của frame
        start_x = max(0, start_x)
        end_x = min(frame.shape[1], end_x)
        start_y = max(0, start_y)
        end_y = min(frame.shape[0], end_y)

        # Cắt vùng cánh tay từ frame
        arm_box = frame[start_y:end_y, start_x:end_x]

        return arm_box

    except Exception as e:
        logger.error(f"Error in extract_arm_region: {e}, start_point: {start_point}, end_point: {end_point}")
        return None  # Trả về None nếu có lỗi

def extract_body_parts_from_frame(frame, person_keypoints):
    """
    Cắt các vùng cơ thể từ frame dựa trên person_keypoints và COCO_EDGES.
    Trả về một dictionary chứa các vùng đã cắt.
    """
    body_parts = {}
    
    # Cắt vùng đầu
    body_parts['head'] = extract_head_from_frame(frame, person_keypoints)

    # Cắt vùng khuỷu tay phải đến cổ tay phải
    right_elbow = person_keypoints[8]  # Khuỷu tay phải
    right_wrist = person_keypoints[10]  # Cổ tay phải
    body_parts['right_arm'] = extract_arm_region(frame, right_elbow, right_wrist)

    # Cắt vùng khuỷu tay trái đến cổ tay trái
    left_elbow = person_keypoints[7]  # Khuỷu tay trái
    left_wrist = person_keypoints[9]  # Cổ tay trái
    body_parts['left_arm'] = extract_arm_region(frame, left_elbow, left_wrist)

    return body_parts