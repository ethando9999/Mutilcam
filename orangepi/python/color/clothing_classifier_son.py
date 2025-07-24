# file: python/clothing_classifier.py (Cập nhật logic theo file mới)

import cv2
import numpy as np
import pandas as pd
import json
import csv
from typing import Optional, List, Dict, Any

from utils.logging_python_orangepi import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

class ClothingClassifier:
    """
    [PHIÊN BẢN CẬP NHẬT THEO LOGIC MỚI]
    - Sử dụng skin tone palette từ CSV
    - Logic phân loại đơn giản hóa: so sánh màu giữa các vùng cơ thể
    - Sử dụng YCrCb color space cho skin detection
    """
    def __init__(
        self,
        skin_csv_path: str,
        sleeve_color_similarity_threshold: float = 10.0,  # Ngưỡng thấp cho áo
        pants_color_similarity_threshold: float = 40.0,   # Ngưỡng cao cho quần
    ):
        self.sleeve_threshold = sleeve_color_similarity_threshold
        self.pants_threshold = pants_color_similarity_threshold
        
        # Load skin tone palette từ CSV
        self.skin_tone_palette = self._load_skin_tone_palette(skin_csv_path)
        
        # Skin detection constants (YCrCb color space)
        self.SKIN_LOWER_BOUND = np.array([0, 133, 77])
        self.SKIN_UPPER_BOUND = np.array([255, 173, 127])
        self.MIN_SKIN_PIXELS = 50

    def _load_skin_tone_palette(self, csv_path: str) -> List[Dict]:
        """Tải bảng màu da từ file CSV (format tương tự file mới)."""
        try:
            palette = []
            with open(csv_path, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    try:
                        # Chuyển từ RGB sang BGR
                        bgr_color = (int(row['b']), int(row['g']), int(row['r']))
                        palette.append({'id': int(row['id']), 'bgr': bgr_color})
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Bỏ qua dòng không hợp lệ: {row}. Lỗi: {e}")
            
            if not palette:
                logger.warning("Không tải được skin palette, sử dụng mặc định")
                return [
                    {'id': 1, 'bgr': (145, 169, 210)},
                    {'id': 2, 'bgr': (130, 175, 225)}
                ]
            
            logger.info(f"Đã tải {len(palette)} tone màu da từ {csv_path}")
            return palette
            
        except Exception as e:
            logger.error(f"Lỗi tải skin palette: {e}")
            return [{'id': 1, 'bgr': (145, 169, 210)}]

    def _are_colors_similar(self, color1_bgr: List[int], color2_bgr: List[int], threshold: float) -> bool:
        """So sánh độ tương tự giữa 2 màu trong không gian LAB."""
        if color1_bgr is None or color2_bgr is None:
            return False
            
        try:
            color1_uint8 = np.uint8([[color1_bgr]])
            color2_uint8 = np.uint8([[color2_bgr]])
            color1_lab = cv2.cvtColor(color1_uint8, cv2.COLOR_BGR2LAB)[0][0]
            color2_lab = cv2.cvtColor(color2_uint8, cv2.COLOR_BGR2LAB)[0][0]
            distance = np.linalg.norm(color1_lab.astype(float) - color2_lab.astype(float))
            return distance < threshold
        except Exception:
            return False

    def _extract_skin_from_forearms(self, regional_data: Dict) -> Optional[List[int]]:
        """
        Trích xuất màu da từ vùng cẳng tay sử dụng YCrCb color space.
        Logic tương tự class PoseColorSignatureExtractor_SkinFiltered trong file mới.
        """
        forearm_colors = regional_data.get("forearm_colors")
        if not forearm_colors:
            return None
            
        # Lấy màu chủ đạo của cẳng tay
        main_forearm_color = forearm_colors[0]["bgr"]
        
        # Kiểm tra xem màu này có phải màu da không bằng YCrCb
        try:
            color_uint8 = np.uint8([[main_forearm_color]])
            color_ycrcb = cv2.cvtColor(color_uint8, cv2.COLOR_BGR2YCrCb)[0][0]
            
            # Kiểm tra trong khoảng skin tone
            is_skin = (self.SKIN_LOWER_BOUND[0] <= color_ycrcb[0] <= self.SKIN_UPPER_BOUND[0] and
                      self.SKIN_LOWER_BOUND[1] <= color_ycrcb[1] <= self.SKIN_UPPER_BOUND[1] and
                      self.SKIN_LOWER_BOUND[2] <= color_ycrcb[2] <= self.SKIN_UPPER_BOUND[2])
            
            return main_forearm_color if is_skin else None
            
        except Exception:
            return None

    def _find_closest_skin_tone(self, detected_bgr_color: List[int]) -> tuple:
        """Tìm tone màu da gần nhất trong palette."""
        if detected_bgr_color is None:
            return None, None

        try:
            detected_lab = cv2.cvtColor(np.uint8([[detected_bgr_color]]), cv2.COLOR_BGR2LAB)[0][0]
            min_dist = float('inf')
            closest_tone = None

            for tone in self.skin_tone_palette:
                palette_bgr = tone['bgr']
                palette_lab = cv2.cvtColor(np.uint8([[palette_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
                dist = np.linalg.norm(detected_lab.astype(float) - palette_lab.astype(float))
                if dist < min_dist:
                    min_dist = dist
                    closest_tone = tone

            return closest_tone['id'], closest_tone['bgr']
        except Exception:
            return None, None

    def classify(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        [LOGIC MỚI - ĐƠN GIẢN HÓA]  
        Phân loại theo logic của file mới: so sánh màu giữa các vùng cơ thể.
        """
        regional = analysis_data.get("regional_analysis", {})
        
        def get_main_color(color_list):
            return color_list[0]["bgr"] if color_list else None

        # Lấy màu chủ đạo từ các vùng
        torso_color = get_main_color(regional.get("torso_colors"))
        forearm_color = get_main_color(regional.get("forearm_colors"))
        thigh_color = get_main_color(regional.get("thigh_colors"))
        shin_color = get_main_color(regional.get("shin_colors"))

        # --- PHÂN LOẠI ÁO (Logic đơn giản như file mới) ---
        sleeve_type = "KHONG THE XAC DINH"
        if torso_color and forearm_color:
            if self._are_colors_similar(torso_color, forearm_color, self.sleeve_threshold):
                sleeve_type = "AO DAI TAY"
            else:
                sleeve_type = "AO NGAN TAY"

        # --- PHÂN LOẠI QUẦN (Logic đơn giản như file mới) ---
        pants_type = "KHONG THE XAC DINH"
        if thigh_color and shin_color:
            if self._are_colors_similar(thigh_color, shin_color, self.pants_threshold):
                pants_type = "QUAN DAI"
            else:
                pants_type = "QUAN NGAN"

        # --- XỬ LÝ MÀU DA (Chỉ khi áo ngắn tay) ---
        skin_tone_bgr = None
        skin_tone_id = None
        
        if sleeve_type == "AO NGAN TAY":
            detected_skin = self._extract_skin_from_forearms(regional)
            if detected_skin:
                skin_tone_id, skin_tone_bgr = self._find_closest_skin_tone(detected_skin)

        return {
            "sleeve_type": sleeve_type,
            "pants_type": pants_type,
            "skin_tone_bgr": skin_tone_bgr,
            "skin_tone_id": skin_tone_id,
            "skin_detection_flags": {
                "forearm_skin_detected": detected_skin is not None if sleeve_type == "AO NGAN TAY" else False
            }
        }