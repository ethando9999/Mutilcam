# file: python/pose_color_signature_son.py (Đã tích hợp logic K-Means)

import asyncio
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple, Dict, Any
from clothing_classifier_son import ClothingClassifier
from utils.logging_python_orangepi import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# =================================================================================
# BƯỚC 1: TÍCH HỢP TOÀN BỘ LOGIC TỪ shirt_color.py VÀO ĐÂY
# =================================================================================
class ColorDetector:
    """
    Lớp này được lấy từ file shirt_color.py của bạn và tích hợp trực tiếp.
    Nó tìm, lọc và gộp các màu chủ đạo trong một vùng ảnh.
    """
    def __init__(self, k=4):
        self.k = k
        self.dominant_colors = None
        self.color_percentages = None

    def find_dominant_colors(self, pixel_data, min_percentage=7.0):
        """Tìm các màu chủ đạo ban đầu bằng KMeans."""
        if pixel_data is None or len(pixel_data) < self.k:
            logger.debug("Không đủ pixel để phân tích màu.")
            return None
            
        # Chuyển đổi sang RGB để tính toán khoảng cách màu trong không gian LAB tốt hơn
        # KMeans thường hoạt động tốt hơn trong không gian màu nhận thức như LAB
        pixels_rgb = cv2.cvtColor(np.uint8([pixel_data]), cv2.COLOR_BGR2RGB)[0]
        
        kmeans = KMeans(n_clusters=self.k, n_init='auto', random_state=42)
        kmeans.fit(pixels_rgb)
        
        total_pixels = len(kmeans.labels_)
        counts = np.bincount(kmeans.labels_)
        
        # Các tâm cụm là màu RGB, chuyển về BGR để sử dụng trong hệ thống
        all_colors_rgb = kmeans.cluster_centers_
        all_colors_bgr = cv2.cvtColor(np.uint8([all_colors_rgb]), cv2.COLOR_RGB2BGR)[0]

        final_colors = []
        final_percentages = []
        sorted_indices = np.argsort(counts)[::-1]

        for i in sorted_indices:
            percentage = (counts[i] / total_pixels) * 100
            if percentage >= min_percentage:
                final_colors.append(all_colors_bgr[i])
                final_percentages.append(percentage)
        
        if not final_colors:
            return None
            
        self.dominant_colors = np.array(final_colors)
        self.color_percentages = np.array(final_percentages)
        return self.dominant_colors

    def merge_similar_colors(self, threshold=50.0):
        """[LOGIC GIỮ NGUYÊN] Gộp các màu tương tự dựa trên ngưỡng trong không gian LAB."""
        if self.dominant_colors is None or len(self.dominant_colors) < 2:
            return

        colors = self.dominant_colors.tolist()
        percentages = self.color_percentages.tolist()

        i = 0
        while i < len(colors):
            j = i + 1
            while j < len(colors):
                # Chuyển đổi màu BGR sang LAB để so sánh
                color1_lab = cv2.cvtColor(np.uint8([[colors[i]]]), cv2.COLOR_BGR2LAB)[0][0]
                color2_lab = cv2.cvtColor(np.uint8([[colors[j]]]), cv2.COLOR_BGR2LAB)[0][0]
                distance = np.linalg.norm(color1_lab.astype(float) - color2_lab.astype(float))

                if distance < threshold:
                    total_percentage = percentages[i] + percentages[j]
                    
                    # Tính trung bình có trọng số cho màu mới
                    new_color = (
                        (np.array(colors[i]) * percentages[i] + np.array(colors[j]) * percentages[j]) / total_percentage
                    )
                    colors[i] = new_color.tolist()
                    percentages[i] = total_percentage
                    
                    # Xóa màu đã được gộp
                    colors.pop(j)
                    percentages.pop(j)
                    j -= 1
                j += 1
            i += 1
        
        # Sắp xếp lại danh sách sau khi gộp
        if colors:
            combined = sorted(zip(percentages, colors), key=lambda x: x[0], reverse=True)
            self.color_percentages, self.dominant_colors = map(np.array, zip(*combined))


    def finalize_results(self, monochromatic_threshold=80.0):
        """[LOGIC GIỮ NGUYÊN] Nếu một màu chiếm ưu thế, chỉ giữ lại màu đó."""
        if self.dominant_colors is None or len(self.dominant_colors) == 0:
            return
        if self.color_percentages[0] > monochromatic_threshold:
            self.dominant_colors = np.array([self.dominant_colors[0]])
            self.color_percentages = np.array([self.color_percentages[0]])

# =================================================================================
# BƯỚC 2: CẬP NHẬT PoseColorAnalyzer ĐỂ SỬ DỤNG LOGIC MỚI
# =================================================================================
class PoseColorAnalyzer:
    """
    [PHIÊN BẢN TÍCH HỢP]
    - Sử dụng ColorDetector (K-Means) để trích xuất NHIỀU màu chủ đạo cho vùng thân (áo).
    - Các vùng khác vẫn giữ logic lấy màu trung bình để đơn giản hóa.
    """
    TORSO_KEYPOINTS: List[int] = [5, 6, 12, 11]
    FOREARM_EDGES: List[Tuple[int, int]] = [(7, 9), (8, 10)]
    THIGH_EDGES: List[Tuple[int, int]] = [(11, 13), (12, 14)]
    SHIN_EDGES: List[Tuple[int, int]] = [(13, 15), (14, 16)]

    def __init__(self, line_thickness: int = 30):
        self.line_thickness = line_thickness
        self.MIN_PIXELS_FOR_COLOR = 50 # Tăng số pixel tối thiểu cho K-Means
        self.LINE_THICKNESS_RATIO = 0.2
        self.MIN_LINE_THICKNESS = 5

    def _is_valid_point(self, point) -> bool:
        return point is not None and len(point) >= 2 and point[0] != 0 and point[1] != 0

    def _extract_dominant_colors_from_torso(self, image: np.ndarray, keypoints: np.ndarray) -> Optional[List[Dict]]:
        """
        [LOGIC MỚI CHO ÁO] Trích xuất nhiều màu chủ đạo từ vùng torso sử dụng ColorDetector.
        """
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            points = [tuple(map(int, keypoints[i][:2])) for i in self.TORSO_KEYPOINTS if i < len(keypoints) and self._is_valid_point(keypoints[i])]
            if len(points) < 3: return None
                
            cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)
            pixels = image[mask == 255]

            if len(pixels) < self.MIN_PIXELS_FOR_COLOR: return None
            
            # 1. Khởi tạo ColorDetector
            detector = ColorDetector(k=5)
            
            # 2. Chạy pipeline phân tích màu từ file shirt_color.py
            detector.find_dominant_colors(pixels, min_percentage=7.0)
            detector.merge_similar_colors(threshold=50.0)
            detector.finalize_results(monochromatic_threshold=80.0)

            if detector.dominant_colors is None: return None

            # 3. Định dạng lại kết quả theo cấu trúc yêu cầu
            # [{ "bgr": [b,g,r], "percentage": p1 }, { "bgr": [b,g,r], "percentage": p2 }]
            results = []
            for color, percentage in zip(detector.dominant_colors, detector.color_percentages):
                results.append({
                    "bgr": [int(c) for c in color],
                    "percentage": round(percentage, 2)
                })
            return results

        except Exception as e:
            logger.debug(f"Lỗi trích xuất màu đa sắc từ polygon: {e}")
            return None

    # --- Các hàm trích xuất màu khác (cho tay, chân) giữ nguyên logic cũ ---
    def _get_mean_color_from_line(self, image: np.ndarray, start_point: tuple, end_point: tuple) -> Optional[List[int]]:
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.line(mask, start_point, end_point, 255, self.line_thickness)
            pixels = image[mask == 255]
            if len(pixels) < 10: return None
            return np.mean(pixels, axis=0).astype(int).tolist()
        except Exception:
            return None

    def _extract_colors_from_edges(self, image: np.ndarray, keypoints: np.ndarray, edges: List[Tuple[int, int]]) -> Optional[List[Dict]]:
        all_colors = []
        for start_idx, end_idx in edges:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = tuple(map(int, keypoints[start_idx][:2]))
                end_point = tuple(map(int, keypoints[end_idx][:2]))
                if self._is_valid_point(start_point) and self._is_valid_point(end_point):
                    color = self._get_mean_color_from_line(image, start_point, end_point)
                    if color:
                        all_colors.append(color)
        
        if not all_colors: return None
        
        mean_color = np.mean(all_colors, axis=0).astype(int).tolist()
        # Trả về theo định dạng list chứa 1 màu duy nhất để tương thích
        return [{"bgr": mean_color, "percentage": 100.0}]


    async def process_and_classify(
        self, image: np.ndarray, keypoints: np.ndarray, classifier: ClothingClassifier, external_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        if external_data is None: external_data = {}
            
        try:
            # [THAY ĐỔI] Gọi hàm trích xuất đa màu cho torso (áo)
            torso_colors = await asyncio.to_thread(
                self._extract_dominant_colors_from_torso, image, keypoints
            )
            # Giữ nguyên logic cũ cho các phần khác
            forearm_colors = await asyncio.to_thread(
                self._extract_colors_from_edges, image, keypoints, self.FOREARM_EDGES  
            )
            thigh_colors = await asyncio.to_thread(
                self._extract_colors_from_edges, image, keypoints, self.THIGH_EDGES
            )
            shin_colors = await asyncio.to_thread(
                self._extract_colors_from_edges, image, keypoints, self.SHIN_EDGES
            )

            regional_analysis = {
                "torso_colors": torso_colors,
                "forearm_colors": forearm_colors, 
                "thigh_colors": thigh_colors, 
                "shin_colors": shin_colors,
            } 

            classifier_input = {**external_data, "regional_analysis": regional_analysis}
            classification_result = await asyncio.to_thread(classifier.classify, classifier_input)
            
            return {
                "classification": classification_result,
                "raw_color_data": regional_analysis
            }

        except Exception as e: 
            logger.error(f"Lỗi trong pipeline phân tích màu sắc tích hợp: {e}", exc_info=True)
            return None