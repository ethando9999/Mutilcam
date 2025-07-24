import asyncio
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from clothing_classifier import ClothingClassifier
# ==============================================================================
# LỚP PHÂN TÍCH MÀU SẮC TỐI ƯU
# ==============================================================================
class PoseColorAnalyzer:
    TORSO_KEYPOINTS: List[int] = [5, 6, 12, 11]
    THIGH_EDGES: List[Tuple[int, int]] = [(11, 13), (12, 14)]
    SHIN_EDGES: List[Tuple[int, int]] = [(13, 15), (14, 16)]

    def __init__(self, k_per_region: int = 5, thickness: int = 30,
                 min_pct: float = 5.0, merge_threshold: float = 30.0, mono_threshold: float = 85.0):
        # (Các tham số giữ nguyên)
        self.k_region = k_per_region
        self.thickness = thickness
        self.min_percentage = min_pct
        self.merge_threshold = merge_threshold
        self.mono_threshold = mono_threshold
    
    # ... Các hàm _extract_pixels, _extract_combined_pixels, _find_dominant_colors, 
    # _merge_similar_colors, _finalize_results, _post_process_colors giữ nguyên như phiên bản trước
    # (Tôi sẽ ẩn chúng đi cho ngắn gọn, nhưng chúng vẫn nằm trong lớp này)
    def _extract_pixels(self, image: np.ndarray, keypoints: np.ndarray, shape: str, indices: Tuple) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if shape == "line":
            start, end = indices
            if not (start < len(keypoints) and end < len(keypoints)): return None
            pt1, pt2 = tuple(map(int, keypoints[start][:2])), tuple(map(int, keypoints[end][:2]))
            if not ((0 <= pt1[0] < w and 0 <= pt1[1] < h) and (0 <= pt2[0] < w and 0 <= pt2[1] < h)): return None
            cv2.line(mask, pt1, pt2, 1, self.thickness)
        elif shape == "poly":
            points = np.array([keypoints[i][:2] for i in indices if i < len(keypoints)], dtype=np.int32)
            if len(points) < 3 or np.any(points < 0): return None
            cv2.fillConvexPoly(mask, points, 1)
        else: return None
        return image[mask > 0]

    async def _extract_combined_pixels(self, image: np.ndarray, keypoints: np.ndarray, edges: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        pixel_tasks = [asyncio.to_thread(self._extract_pixels, image, keypoints, "line", edge) for edge in edges]
        pixel_groups = await asyncio.gather(*pixel_tasks)
        valid_pixel_groups = [p for p in pixel_groups if p is not None and len(p) > 0]
        if not valid_pixel_groups: return None
        return np.concatenate(valid_pixel_groups)

    def _find_dominant_colors(self, pixels: np.ndarray, k: int) -> Optional[Dict[str, np.ndarray]]:
        if pixels is None or len(pixels) < k: return None
        pixels_lab_all = cv2.cvtColor(pixels.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        lightness = pixels_lab_all[:, 0]
        valid_indices = (lightness >= 10) & (lightness <= 95)
        data_to_cluster = pixels_lab_all[valid_indices].astype(np.float32)
        if len(data_to_cluster) < k: return None
        kmeans = KMeans(n_clusters=min(k, len(data_to_cluster)), n_init='auto', random_state=42)
        kmeans.fit(data_to_cluster)
        counts = np.bincount(kmeans.labels_)
        total_pixels = len(data_to_cluster)
        sorted_indices = np.argsort(counts)[::-1]
        final_lab, final_pct = [], []
        for i in sorted_indices:
            percentage = (counts[i] / total_pixels) * 100
            if percentage >= self.min_percentage:
                final_lab.append(kmeans.cluster_centers_[i])
                final_pct.append(percentage)
        if not final_lab: return None
        final_lab_np = np.array(final_lab, dtype=np.float32)
        final_bgr_np = cv2.cvtColor(final_lab_np.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        return {"colors_lab": final_lab_np, "colors_bgr": final_bgr_np, "percentages": np.array(final_pct, dtype=np.float32)}

    def _merge_similar_colors(self, color_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if color_data is None or len(color_data["colors_lab"]) < 2: return color_data
        labs, bgrs, pcts = color_data["colors_lab"].tolist(), color_data["colors_bgr"].tolist(), color_data["percentages"].tolist()
        i = 0
        while i < len(labs):
            j = i + 1
            while j < len(labs):
                dist = np.linalg.norm(np.array(labs[i]) - np.array(labs[j]))
                if dist < self.merge_threshold:
                    total_pct = pcts[i] + pcts[j]
                    new_lab = ((np.array(labs[i]) * pcts[i] + np.array(labs[j]) * pcts[j])) / total_pct
                    labs[i], pcts[i] = new_lab.tolist(), total_pct
                    bgrs[i] = cv2.cvtColor(np.uint8([[new_lab]]), cv2.COLOR_LAB2BGR)[0][0].tolist()
                    labs.pop(j); bgrs.pop(j); pcts.pop(j)
                else: j += 1
            i += 1
        combined = sorted(zip(pcts, bgrs, labs), key=lambda x: x[0], reverse=True)
        s_pcts, s_bgrs, s_labs = zip(*combined)
        return {"colors_lab": np.array(s_labs), "colors_bgr": np.array(s_bgrs), "percentages": np.array(s_pcts)}

    def _finalize_results(self, color_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if color_data is None or len(color_data["percentages"]) == 0: return color_data
        if color_data["percentages"][0] > self.mono_threshold:
            return {k: np.array([v[0]]) for k, v in color_data.items()}
        return color_data

    def _post_process_colors(self, color_data: Optional[Dict]) -> Optional[List[Dict[str, Any]]]:
        if color_data is None: return None
        merged_data = self._merge_similar_colors(color_data)
        final_data = self._finalize_results(merged_data)
        return [{"bgr": bgr.astype(int).tolist(), "percentage": round(pct, 2)}
                for bgr, pct in zip(final_data["colors_bgr"], final_data["percentages"])]

    async def process_and_classify(
        self, image: np.ndarray, keypoints: np.ndarray, classifier: ClothingClassifier, external_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        [TỐI ƯU] Pipeline chính: Trích xuất màu -> Hậu xử lý -> Gọi Classifier.
        """
        if external_data is None: external_data = {}
        try:
            # [CẢI TIẾN] Thêm phân tích bắp chân (shin) để phân biệt quần dài/ngắn
            pixel_tasks = {
                "torso": asyncio.to_thread(self._extract_pixels, image, keypoints, "poly", self.TORSO_KEYPOINTS),
                "thigh": self._extract_combined_pixels(image, keypoints, self.THIGH_EDGES),
                "shin": self._extract_combined_pixels(image, keypoints, self.SHIN_EDGES),
            }
            pixel_results = await asyncio.gather(*pixel_tasks.values())
            pixel_map = dict(zip(pixel_tasks.keys(), pixel_results))

            analysis_tasks = {
                "torso_raw": asyncio.to_thread(self._find_dominant_colors, pixel_map["torso"], self.k_region),
                "thigh_raw": asyncio.to_thread(self._find_dominant_colors, pixel_map["thigh"], self.k_region),
                "shin_raw": asyncio.to_thread(self._find_dominant_colors, pixel_map["shin"], self.k_region),
            }
            raw_color_results = await asyncio.gather(*analysis_tasks.values())
            raw_color_map = dict(zip(analysis_tasks.keys(), raw_color_results))

            post_process_tasks = {
                "torso_final": asyncio.to_thread(self._post_process_colors, raw_color_map["torso_raw"]),
                "thigh_final": asyncio.to_thread(self._post_process_colors, raw_color_map["thigh_raw"]),
                "shin_final": asyncio.to_thread(self._post_process_colors, raw_color_map["shin_raw"]),
            }
            final_color_results = await asyncio.gather(*post_process_tasks.values())
            final_color_map = dict(zip(post_process_tasks.keys(), final_color_results))
            
            # Chuẩn bị dữ liệu cho classifier
            classifier_input = {
                **external_data,
                "regional_analysis": {
                    "torso_colors": final_color_map["torso_final"],
                    "thigh_colors": final_color_map["thigh_final"],
                    "shin_colors": final_color_map["shin_final"],
                }
            }
            
            # Chạy classifier và trả về kết quả cuối cùng
            classification_result = await asyncio.to_thread(classifier.classify, classifier_input)
            
            return {
                "classification": classification_result,
                "raw_color_data": classifier_input["regional_analysis"]
            }

        except Exception as e:
            print(f"ERROR: An error occurred in the analysis pipeline: {e}")
            return None

# ==============================================================================
# HÀM TIỆN ÍCH VÀ THỰC THI CHÍNH
# ==============================================================================
def show_color_palette(color_data: Optional[List[Dict]], title: str):
    """Hàm tiện ích để hiển thị bảng màu."""
    if not color_data: return
    sorted_data = sorted(color_data, key=lambda x: x['percentage'], reverse=True)
    percentages = [item['percentage'] for item in sorted_data]
    colors_rgb = [np.array(item['bgr'])[::-1] for item in sorted_data] # BGR to RGB
    palette = np.zeros((50, 300, 3), dtype='uint8')
    start_x = 0
    total_percentage = sum(percentages)
    if total_percentage == 0: return
    for i, p in enumerate(percentages):
        color_width = int((p / total_percentage) * 300)
        end_x = start_x + color_width
        cv2.rectangle(palette, (start_x, 0), (end_x, 50), colors_rgb[i].tolist(), -1)
        start_x = end_x
    plt.figure(title, figsize=(6, 2)); plt.title(title); plt.axis("off"); plt.imshow(palette)

# ==============================================================================
# HÀM TIỆN ÍCH VÀ THỰC THI CHÍNH
# ==============================================================================
def show_color_palette(color_data: Optional[List[Dict]], title: str):
    """
    Hàm tiện ích để hiển thị bảng màu từ kết quả phân tích.
    """
    if not color_data:
        print(f"\nNo color data to display for '{title}'.")
        return

    print(f"\n--- {title} Color Palette ---")
    
    # Sắp xếp để hiển thị nhất quán
    sorted_data = sorted(color_data, key=lambda x: x['percentage'], reverse=True)
    
    percentages = [item['percentage'] for item in sorted_data]
    # Đầu vào là BGR, Matplotlib cần RGB, nên phải đảo ngược kênh màu
    colors_rgb = [np.array(item['bgr'])[::-1] for item in sorted_data]
    
    for i, item in enumerate(sorted_data):
        print(f"- Color (BGR): {item['bgr']}, Coverage: {item['percentage']:.2f}%")

    palette = np.zeros((50, 300, 3), dtype='uint8')
    start_x = 0
    total_percentage = sum(percentages)
    if total_percentage == 0: return

    for i, p in enumerate(percentages):
        color_width = int((p / total_percentage) * 300)
        end_x = start_x + color_width
        
        # Vẽ hình chữ nhật với màu RGB
        cv2.rectangle(palette, (start_x, 0), (end_x, 50), colors_rgb[i].tolist(), -1)
        start_x = end_x

    plt.figure(title, figsize=(6, 2))
    plt.title(title)
    plt.axis("off")
    plt.imshow(palette)