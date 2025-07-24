# file: python/clothing_classifier.py

import cv2
import numpy as np
import pandas as pd
import json
from typing import Optional, List, Dict, Any

class ClothingClassifier:
    """
    [PHIÊN BẢN TỐI ƯU HOÀN CHỈNH]
    - Sửa lỗi tràn số (overflow) khi tính toán trong không gian màu LAB.
    - Thêm tính năng tính toán màu da trung bình từ các vùng da hở.
    - Tăng cường khả năng chống lỗi với các dữ liệu đầu vào có thể bị thiếu.
    - Cung cấp output có cấu trúc, sẵn sàng cho việc tạo payload.
    """
    def __init__(
        self,
        skin_csv_path: str,
        skin_color_threshold: float = 38.0,
        clothing_color_similarity_threshold: float = 35.0,
        achromatic_threshold: float = 20.0
    ):
        self.skin_color_threshold = skin_color_threshold
        self.clothing_color_similarity_threshold = clothing_color_similarity_threshold
        self.achromatic_threshold = achromatic_threshold
        
        # Tải bộ mẫu màu da từ file CSV
        self.skin_tone_samples_bgr = self._load_skin_tones_from_csv(skin_csv_path)
        
        # Chuyển đổi bộ mẫu sang không gian màu LAB một lần duy nhất để tối ưu hiệu suất
        self.skin_tone_samples_lab = cv2.cvtColor(
            self.skin_tone_samples_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3)

    def _load_skin_tones_from_csv(self, csv_path: str) -> np.ndarray:
        """Đọc và phân tích cú pháp màu da từ file CSV một cách an toàn."""
        try:
            df = pd.read_csv(csv_path)
            skin_tones = []
            for col in df.columns:
                for color_str in df[col].dropna():
                    try:
                        color_list = json.loads(color_str)
                        if isinstance(color_list, list) and len(color_list) == 3:
                            skin_tones.append(color_list)
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            print(f"INFO: Successfully loaded {len(skin_tones)} skin tone samples from '{csv_path}'.")
            return np.array(skin_tones, dtype=np.uint8)
        
        except FileNotFoundError:
            print(f"WARNING: Skin tone CSV not found at '{csv_path}'. Using a small default set.")
            return np.array([
                [145, 169, 210], [130, 175, 225], [110, 140, 190],
                [80, 110, 160],  [193, 165, 160], [211, 229, 240]
            ], dtype=np.uint8)

    def _to_lab(self, bgr: List[int]) -> Optional[np.ndarray]:
        """Chuyển đổi một màu BGR sang không gian LAB."""
        if bgr is None: return None
        return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

    def _color_distance(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Tính khoảng cách Euclidean giữa hai màu trong không gian LAB."""
        return np.linalg.norm(lab1.astype(float) - lab2.astype(float))

    def is_skin_tone(self, bgr_color: List[int]) -> bool:
        """Kiểm tra một màu có phải là màu da hay không."""
        if bgr_color is None: return False
        
        lab_color = self._to_lab(bgr_color)
        if lab_color is None: return False

        # [SỬA LỖI] Chuyển đổi sang float TRƯỚC KHI trừ để tránh tràn số (overflow).
        # `uint8(10) - 128` sẽ bị tràn, `float(10) - 128` thì không.
        chroma = np.hypot(float(lab_color[1]) - 128, float(lab_color[2]) - 128)
        
        # Loại bỏ các màu vô sắc (trắng, xám, đen)
        if chroma < self.achromatic_threshold:
            return False
            
        # Tìm khoảng cách ngắn nhất đến một mẫu da trong bộ tham chiếu
        min_dist = min(self._color_distance(lab_color, s_lab) for s_lab in self.skin_tone_samples_lab)
        
        return min_dist < self.skin_color_threshold

    def classify(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        [TỐI ƯU] Pipeline phân loại chính, trả về kết quả có cấu trúc.
        """
        regional = analysis_data.get("regional_analysis", {})
        gender = analysis_data.get("gender_analysis", {}).get("gender", "unknown").lower()

        # --- Lấy màu chủ đạo và kiểm tra có phải màu da không ---
        def get_main_color(color_list):
            return color_list[0]["bgr"] if color_list else None

        # --- Lấy màu chủ đạo và kiểm tra có phải màu da không ---
        torso_color = get_main_color(regional.get("torso_colors"))
        # << THAY ĐỔI: Lấy thêm màu cẳng tay >>
        forearm_color = get_main_color(regional.get("forearm_colors")) 
        thigh_color = get_main_color(regional.get("thigh_colors"))
        shin_color = get_main_color(regional.get("shin_colors"))

        is_skin_torso = self.is_skin_tone(torso_color)
        is_skin_forearm = self.is_skin_tone(forearm_color) # << THAY ĐỔI
        is_skin_thigh = self.is_skin_tone(thigh_color)
        is_skin_shin = self.is_skin_tone(shin_color)

        # << THAY ĐỔI: Cập nhật logic phân loại áo >>
        sleeve_type = "Không phát hiện áo"
        if torso_color and not is_skin_torso:
            # Nếu có áo, kiểm tra cẳng tay để xác định độ dài
            if forearm_color and not is_skin_forearm:
                sleeve_type = "Áo tay dài"
            elif is_skin_forearm:
                sleeve_type = "Áo tay ngắn"
            else: # Không có thông tin cẳng tay
                sleeve_type = "Áo (không rõ độ dài tay)"
        elif is_skin_torso:
             sleeve_type = "Không xác định đc áo (hoặc áo màu da)"

        # --- Phân loại Quần / Váy (Logic giữ nguyên) ---
        pants_type = "Không phát hiện quần/váy"
        if thigh_color and not is_skin_thigh: 
            if is_skin_shin:
                pants_type = "Váy hoặc Quần đùi" if gender == "female" else "Quần đùi"
            elif shin_color and not is_skin_shin:
                dist = self._color_distance(self._to_lab(thigh_color), self._to_lab(shin_color))
                pants_type = "Quần dài" if dist < self.clothing_color_similarity_threshold else "Quần dài (khác màu) hoặc Váy dài"
            else:
                pants_type = "Quần (không rõ độ dài)"
        elif is_skin_thigh:
            pants_type = "Không xác định (hoặc trang phục rất ngắn)"

        # --- [TÍNH NĂNG MỚI] Tính toán màu da trung bình ---
        detected_skin_colors = []
        if is_skin_torso: detected_skin_colors.append(torso_color)
        if is_skin_thigh: detected_skin_colors.append(thigh_color)
        if is_skin_shin: detected_skin_colors.append(shin_color)
        
        avg_skin_tone_bgr = None
        if detected_skin_colors:
            avg_skin_tone_bgr = np.round(np.mean(detected_skin_colors, axis=0)).astype(int).tolist()

        # --- Trả về kết quả cuối cùng có cấu trúc ---
        # --- Trả về kết quả cuối cùng có cấu trúc ---
        return {
            "sleeve_type": sleeve_type,
            "pants_type": pants_type,
            "skin_tone_bgr": avg_skin_tone_bgr,
            "skin_detection_flags": {
                "torso": is_skin_torso,
                "forearm": is_skin_forearm, # << THAY ĐỔI
                "thigh": is_skin_thigh,
                "shin": is_skin_shin
            }
        }