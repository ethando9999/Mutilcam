# file: python/pose_color_signature.py (Cập nhật logic đơn giản theo file mới)

import asyncio
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from clothing_classifier import ClothingClassifier
from utils.logging_python_orangepi import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

class PoseColorAnalyzer:
    """
    [PHIÊN BẢN ĐƠN GIẢN HÓA THEO FILE MỚI]
    - Logic đơn giản: trích xuất màu chủ đạo từ các vùng cơ thể
    - Bỏ các xử lý phức tạp như merge colors, clustering nhiều màu
    - Tập trung vào việc lấy màu chủ đạo để so sánh
    """
    
    # Định nghĩa các vùng cơ thể (giữ nguyên)
    TORSO_KEYPOINTS: List[int] = [5, 6, 12, 11] # Vai trái, vai phải, hông phải, hông trái
    FOREARM_EDGES: List[Tuple[int, int]] = [(7, 9), (8, 10)] # Khuỷu tay -> Cổ tay (trái, phải)
    THIGH_EDGES: List[Tuple[int, int]] = [(11, 13), (12, 14)] # Hông -> Đầu gối (trái, phải)
    SHIN_EDGES: List[Tuple[int, int]] = [(13, 15), (14, 16)] # Đầu gối -> Mắt cá (trái, phải)

    def __init__(self, line_thickness: int = 30):
        """Đơn giản hóa constructor - chỉ cần line thickness."""
        self.line_thickness = line_thickness
        # Constants từ file mới
        self.MIN_PIXELS_FOR_COLOR = 10
        self.LINE_THICKNESS_RATIO = 0.2
        self.MIN_LINE_THICKNESS = 5

    def _is_valid_point(self, point) -> bool:
        """Kiểm tra điểm có hợp lệ không."""
        return point is not None and len(point) >= 2 and point[0] != 0 and point[1] != 0

    def _get_dominant_color_simple(self, image: np.ndarray, start_point: tuple, end_point: tuple) -> Optional[List[int]]:
        """
        Lấy màu chủ đạo đơn giản từ đường thẳng giữa 2 điểm.
        Logic tương tự _get_dominant_color trong file mới.
        """
        try:
            # Tạo mask cho vùng đường thẳng
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.line(mask, start_point, end_point, 255, self.line_thickness)
            
            # Lấy pixels trong vùng
            pixels = image[mask == 255]
            if len(pixels) < self.MIN_PIXELS_FOR_COLOR:
                return None
                
            # Tính màu trung bình (đơn giản hơn clustering)    
            mean_color = np.mean(pixels, axis=0).astype(int)
            return mean_color.tolist()
            
        except Exception as e:
            logger.debug(f"Lỗi lấy màu dominant: {e}")
            return None

    def _extract_pixels_from_polygon(self, image: np.ndarray, keypoints: np.ndarray, indices: List[int]) -> Optional[List[int]]:
        """Trích xuất màu từ vùng polygon (cho torso)."""
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Tạo polygon từ các keypoints
            points = []
            for i in indices:
                if i < len(keypoints):
                    point = keypoints[i][:2]
                    if self._is_valid_point(point):
                        points.append([int(point[0]), int(point[1])])
            
            if len(points) < 3:
                return None
                
            points_array = np.array(points, dtype=np.int32)
            cv2.fillConvexPoly(mask, points_array, 255)
            
            # Lấy pixels và tính màu trung bình
            pixels = image[mask == 255]
            if len(pixels) < self.MIN_PIXELS_FOR_COLOR:
                return None
                
            mean_color = np.mean(pixels, axis=0).astype(int)
            return mean_color.tolist()
            
        except Exception as e:
            logger.debug(f"Lỗi trích xuất từ polygon: {e}")
            return None

    def _extract_colors_from_edges(self, image: np.ndarray, keypoints: np.ndarray, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
        """
        Trích xuất màu từ nhiều edges và trả về màu trung bình.
        Logic đơn giản hóa từ _extract_combined_pixels.
        """
        try:
            all_colors = []
            
            for start_idx, end_idx in edges:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_point = tuple(map(int, keypoints[start_idx][:2]))
                    end_point = tuple(map(int, keypoints[end_idx][:2]))
                    
                    if self._is_valid_point(start_point) and self._is_valid_point(end_point):
                        # Tính line thickness động
                        line_length = np.linalg.norm(np.array(start_point) - np.array(end_point))
                        dynamic_thickness = int(line_length * self.LINE_THICKNESS_RATIO) + self.MIN_LINE_THICKNESS
                        
                        color = self._get_dominant_color_simple(image, start_point, end_point)
                        if color:
                            all_colors.append(color)
            
            if not all_colors:
                return None
                
            # Trả về màu trung bình của tất cả edges
            mean_color = np.mean(all_colors, axis=0).astype(int)
            return mean_color.tolist()
            
        except Exception as e:
            logger.debug(f"Lỗi trích xuất từ edges: {e}")
            return None

    async def process_and_classify(
        self, image: np.ndarray, keypoints: np.ndarray, classifier: ClothingClassifier, external_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        [PIPELINE ĐƠN GIẢN HÓA] 
        Trích xuất màu chủ đạo từ các vùng -> Gọi classifier đơn giản.
        """
        if external_data is None: 
            external_data = {}
            
        try:
            # Trích xuất màu đơn giản từ 4 vùng cơ thể
            torso_color = await asyncio.to_thread(
                self._extract_pixels_from_polygon, image, keypoints, self.TORSO_KEYPOINTS
            )
            forearm_color = await asyncio.to_thread(
                self._extract_colors_from_edges, image, keypoints, self.FOREARM_EDGES  
            )
            thigh_color = await asyncio.to_thread(
                self._extract_colors_from_edges, image, keypoints, self.THIGH_EDGES
            )
            shin_color = await asyncio.to_thread(
                self._extract_colors_from_edges, image, keypoints, self.SHIN_EDGES
            )

            # Format dữ liệu theo định dạng cũ (để tương thích với classifier)
            def format_color_data(color):
                if color is None:
                    return None
                return [{"bgr": color, "percentage": 100.0}]  # Đơn giản - chỉ 1 màu với 100%

            regional_analysis = {
                "torso_colors": format_color_data(torso_color),
                "forearm_colors": format_color_data(forearm_color), 
                "thigh_colors": format_color_data(thigh_color), 
                "shin_colors": format_color_data(shin_color),
            } 

            # Chuẩn bị dữ liệu cho classifier
            classifier_input = {
                **external_data,
                "regional_analysis": regional_analysis
            }
            
            # Gọi classifier
            classification_result = await asyncio.to_thread(classifier.classify, classifier_input)
            
            return {
                "classification": classification_result,
                "raw_color_data": regional_analysis
            }

        except Exception as e: 
            logger.error(f"Lỗi trong pipeline phân tích màu sắc đơn giản: {e}", exc_info=True)
            return None