# Danh sách các điểm keypoint theo chuẩn COCO
COCO_KEYPOINTS = [
    (0, "Mũi"),
    (1, "Mắt trái"),
    (2, "Mắt phải"),
    (3, "Tai trái"),
    (4, "Tai phải"),
    (5, "Vai trái"),
    (6, "Vai phải"),
    (7, "Khuỷu tay trái"),
    (8, "Khuỷu tay phải"),
    (9, "Cổ tay trái"),
    (10, "Cổ tay phải"),
    (11, "Hông trái"),
    (12, "Hông phải"),
    (13, "Đầu gối trái"),
    (14, "Đầu gối phải"),
    (15, "Mắt cá chân trái"),
    (16, "Mắt cá chân phải")
]
# file: python/core/keypoints_handle.py (Đã tối ưu)

import numpy as np

# --- Các hằng số (Giữ nguyên) ---
NOSE_IDX = 0; LEFT_EYE_IDX = 1; RIGHT_EYE_IDX = 2; LEFT_EAR_IDX = 3; RIGHT_EAR_IDX = 4
LEFT_SHOULDER_IDX = 5; RIGHT_SHOULDER_IDX = 6; LEFT_HIP_IDX = 11; RIGHT_HIP_IDX = 12

# --- Các hàm gốc (Giữ nguyên) ---
def is_valid_kpt(point):
    """Kiểm tra một keypoint có hợp lệ không (tọa độ > 0)."""
    return point is not None and point[0] > 0 and point[1] > 0

def get_head_center(keypoints: list):
    """Lấy vị trí của mũi. Nếu không có, tính trung tâm các điểm khác trên đầu."""
    if is_valid_kpt(keypoints[NOSE_IDX]):
        return tuple(map(int, keypoints[NOSE_IDX][:2]))
    
    head_indices = [LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_EAR_IDX, RIGHT_EAR_IDX]
    valid_points = [keypoints[i][:2] for i in head_indices if is_valid_kpt(keypoints[i])]
    
    if valid_points:
        return tuple(np.mean(valid_points, axis=0).astype(int))
    return None

def get_torso_box(keypoints, full_box):
    """Tạo một bounding box chỉ chứa phần thân người từ keypoint vai và hông."""
    shoulder_indices = [LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX]
    hip_indices = [LEFT_HIP_IDX, RIGHT_HIP_IDX]

    valid_shoulders = [keypoints[i][:2] for i in shoulder_indices if is_valid_kpt(keypoints[i])]
    valid_hips = [keypoints[i][:2] for i in hip_indices if is_valid_kpt(keypoints[i])]
    
    if not valid_shoulders or not valid_hips:
        return full_box

    torso_points = np.array(valid_shoulders + valid_hips)
    
    xmin, ymin = np.min(torso_points, axis=0)
    xmax, ymax = np.max(torso_points, axis=0)

    padding_x = (xmax - xmin) * 0.15
    padding_y = (ymax - ymin) * 0.10

    torso_xmin = max(full_box[0], int(xmin - padding_x))
    torso_ymin = max(full_box[1], int(ymin - padding_y))
    torso_xmax = min(full_box[2], int(xmax + padding_x))
    torso_ymax = min(full_box[3], int(ymax + padding_y))

    return (torso_xmin, torso_ymin, torso_xmax, torso_ymax)

# <<< ======================= HÀM MỚI ĐỂ SỬA LỖI ======================= >>>
def adjust_keypoints_to_box(keypoints: np.ndarray, box: tuple) -> np.ndarray:
    """
    Chuyển đổi tọa độ keypoints từ hệ quy chiếu của frame gốc
    sang hệ quy chiếu của một bounding box đã cắt.

    Args:
        keypoints (np.array): Mảng keypoints gốc (17, 3).
        box (tuple): Bounding box (xmin, ymin, xmax, ymax) đã dùng để cắt.

    Returns:
        np.array: Mảng keypoints mới với tọa độ đã được điều chỉnh.
    """
    adjusted_kpts = keypoints.copy()
    xmin, ymin, _, _ = box
    
    # Chỉ điều chỉnh các keypoint hợp lệ
    valid_kpts_mask = adjusted_kpts[:, :2].sum(axis=1) > 0
    adjusted_kpts[valid_kpts_mask, 0] -= xmin
    adjusted_kpts[valid_kpts_mask, 1] -= ymin
    
    return adjusted_kpts

def get_optimal_feet_point(keypoints: np.ndarray, box: tuple) -> tuple[int, int]:
    """
    Lấy điểm chân tối ưu bằng phương pháp phân tầng.

    Ưu tiên 1: Trung điểm mắt cá chân.
    Ưu tiên 2: Trung điểm đầu gối.
    Ưu tiên 3: Trung điểm cạnh dưới bounding box.

    Args:
        keypoints (np.ndarray): Mảng keypoints của một người.
        box (tuple): Bounding box (x1, y1, x2, y2) của người đó.

    Returns:
        tuple[int, int]: Tọa độ (u, v) của điểm chân tối ưu.
    """
    # Chỉ số keypoints mắt cá chân (15, 16) và đầu gối (13, 14)
    ANKLE_KPS_INDICES = [15, 16]
    KNEE_KPS_INDICES = [13, 14]

    # Lấy tọa độ và điểm tin cậy (nếu có)
    kpts_xy = keypoints[:, :2]
    kpts_conf = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(keypoints.shape[0])

    # Ưu tiên 1: Mắt cá chân
    valid_ankles = kpts_xy[ANKLE_KPS_INDICES][kpts_conf[ANKLE_KPS_INDICES] > 0.3]
    if len(valid_ankles) > 0:
        # Lấy trung bình của các mắt cá chân hợp lệ
        u_feet, v_feet = np.mean(valid_ankles, axis=0)
        return int(u_feet), int(v_feet)

    # Ưu tiên 2: Đầu gối
    valid_knees = kpts_xy[KNEE_KPS_INDICES][kpts_conf[KNEE_KPS_INDICES] > 0.3]
    if len(valid_knees) > 0:
        # Lấy trung bình của các đầu gối hợp lệ
        u_feet, v_feet = np.mean(valid_knees, axis=0)
        return int(u_feet), int(v_feet)

    # Ưu tiên 3: Bounding box
    x1, _, x2, y2 = map(int, box)
    return (x1 + x2) // 2, y2
# <<< ===================== KẾT THÚC HÀM MỚI ===================== >>>