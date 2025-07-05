import numpy as np

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

# Định nghĩa các chỉ số để code dễ đọc hơn
NOSE_IDX = 0
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2
LEFT_EAR_IDX = 3
RIGHT_EAR_IDX = 4

def get_head_center(keypoints: list):
    """
    Lấy vị trí của mũi. Nếu không có mũi, tính toán vị trí trung tâm của
    các điểm khác trên đầu (mắt, tai).

    Args:
        keypoints (list): Một danh sách gồm 17 keypoint. Mỗi keypoint có thể là
                          một tuple (x, y) hoặc (x, y, confidence), hoặc None.
                          Một điểm không được phát hiện thường có tọa độ (0, 0).

    Returns:
        tuple: Tọa độ (x, y) của mũi hoặc trung tâm đầu.
        None: Nếu không có điểm nào trên đầu được phát hiện.
    """
    
    # --- 1. Hàm phụ để kiểm tra một điểm có hợp lệ không ---
    def is_valid(point):
        # Trả về False nếu điểm là None hoặc có tọa độ (0,0)
        if point is None or (point[0] == 0 and point[1] == 0):
            return False
        return True

    # --- 2. Kiểm tra điểm Mũi (Nose) ---
    nose_point = keypoints[NOSE_IDX]
    if is_valid(nose_point):
        # Nếu mũi tồn tại, trả về tọa độ (x, y) của nó
        return (int(nose_point[0]), int(nose_point[1]))

    # --- 3. Nếu Mũi không tồn tại, tính toán trung tâm của các điểm khác trên đầu ---
    else:
        head_parts_indices = [
            LEFT_EYE_IDX,
            RIGHT_EYE_IDX,
            LEFT_EAR_IDX,
            RIGHT_EAR_IDX
        ]
        
        valid_head_points = []
        for idx in head_parts_indices:
            point = keypoints[idx]
            if is_valid(point):
                valid_head_points.append(point)

        # Nếu có các điểm hợp lệ trên đầu
        if valid_head_points:
            # Dùng numpy để tính trung bình cộng các tọa độ một cách dễ dàng
            points_array = np.array(valid_head_points)[:, :2] # Chỉ lấy x, y
            center_point = np.mean(points_array, axis=0)
            return (int(center_point[0]), int(center_point[1]))
        
        # --- 4. Nếu không có điểm nào trên đầu hợp lệ ---
        else:
            return None