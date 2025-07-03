import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import faiss
import time
from .logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Danh sách các cặp keypoint (edges) theo định dạng COCO
COCO_EDGES = [  
    (0, 1), (0, 2), (1, 3), (2, 4),      # Head và facial connections
    (5, 6),                             # Shoulders
    (5, 7), (7, 9),                     # Left arm
    (6, 8), (8, 10),                    # Right arm
    (11, 12),                           # Hips
    (11, 13), (13, 15),                 # Left leg
    (12, 14), (14, 16),                 # Right leg
    (5, 11), (6, 12),                   # Torso
    (6, 11)                            # Body
]
MIN_POINTS_FOR_CLUSTERING = 50

class PoseColorSignatureExtractor:
    COCO_EDGES = COCO_EDGES
    
    def __init__(self, n_cluster=3, d=3, n_iter=50, verbose=False, color_space='BGR', thickness=20):
        self.n_cluster = n_cluster
        self.d = d
        self.n_iter = n_iter
        self.verbose = verbose
        self.color_space = color_space.upper()
        self.thickness = thickness

    def preprocess_pixels(self, pixels):
        if len(pixels) == 0:
            return pixels
        combined_mask = np.ones(len(pixels), dtype=bool)
        for c in range(self.d):
            lower = np.percentile(pixels[:, c], 5)
            upper = np.percentile(pixels[:, c], 95)
            channel_mask = (pixels[:, c] >= lower) & (pixels[:, c] <= upper)
            combined_mask &= channel_mask
        return pixels[combined_mask]

    def extract_edge_pixels_sync(self, image, start, end, keypoints):
        if start >= len(keypoints) or end >= len(keypoints):
            return None
        pt1 = tuple(map(int, keypoints[start][:2]))
        pt2 = tuple(map(int, keypoints[end][:2]))
        if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
            return None
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.line(mask, pt1, pt2, color=1, thickness=self.thickness)
        pixels = image[mask > 0]
        # logger.debug(f"Extracted {len(pixels)} pixels for edge ({start}, {end})")
        return pixels if len(pixels) > 0 else None

    async def extract_edge_pixels_async(self, image, start, end, keypoints):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.extract_edge_pixels_sync, image, start, end, keypoints)

    def find_raw_dominant_color(self, pixels):
        if len(pixels) == 0: 
            return None
        preprocessed = self.preprocess_pixels(pixels)
        # logger.debug(f"After preprocessing: {len(preprocessed)} pixels remain") 
        
        if len(preprocessed) < max(self.n_cluster, MIN_POINTS_FOR_CLUSTERING):
            if len(pixels) > 0:
                # logger.warning(f"Insufficient points for clustering: {len(preprocessed)} < {max(self.n_cluster, MIN_POINTS_FOR_CLUSTERING)}. Using mean color.")
                return np.mean(pixels, axis=0)
            return None
        if self.color_space == 'LAB':
            preprocessed_uint8 = np.clip(preprocessed, 0, 255).astype(np.uint8)
            preprocessed_lab = cv2.cvtColor(preprocessed_uint8.reshape(-1, 1, 3),
                                            cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            data = preprocessed_lab
        else:
            data = preprocessed.astype(np.float32)
        kmeans = faiss.Kmeans(d=self.d, k=self.n_cluster, niter=self.n_iter, verbose=self.verbose)
        kmeans.train(data)
        _, labels = kmeans.index.search(data, 1)
        counts = np.bincount(labels.ravel(), minlength=self.n_cluster)
        dominant_cluster_idx = np.argmax(counts)
        dominant_color = kmeans.centroids[dominant_cluster_idx]
        if self.color_space == 'LAB':
            dominant_color_uint8 = np.clip(dominant_color, 0, 255).astype(np.uint8)
            dominant_color_bgr = cv2.cvtColor(dominant_color_uint8.reshape(1, 1, 3),
                                            cv2.COLOR_LAB2BGR)[0, 0]
            return dominant_color_bgr
        return dominant_color

    async def find_raw_dominant_color_async(self, pixels):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.find_raw_dominant_color, pixels)

    def get_raw_color_signature(self, edge_pixels):
        raw_signature = []
        for pixels in edge_pixels:
            if pixels is None or len(pixels) < self.n_cluster:
                raw_signature.append(None)
                continue
            raw_color = self.find_raw_dominant_color(pixels)
            raw_signature.append(raw_color)
        return raw_signature

    def visualize_color_signature(self, color_signature, stripe_width=50, stripe_height=50, spacing=5):
        num_stripes = len(color_signature)
        width = spacing + num_stripes * (stripe_width + spacing)
        height = stripe_height + 2 * spacing
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        for i, color in enumerate(color_signature):
            x = spacing + i * (stripe_width + spacing)
            y = spacing
            if color is None:
                fill_color = (200, 200, 200)
            else:
                fill_color = tuple(int(round(c)) for c in color)
            cv2.rectangle(img, (x, y), (x + stripe_width, y + stripe_height),
                          fill_color, thickness=-1)
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness_text = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness_text)
            text_x = x + (stripe_width - text_size[0]) // 2
            text_y = y + (stripe_height + text_size[1]) // 2
            brightness = sum(fill_color) if color is not None else 600
            text_color = (0, 0, 0) if brightness > 382 else (255, 255, 255)
            cv2.putText(img, text, (text_x, text_y), font, font_scale,
                        text_color, thickness_text, cv2.LINE_AA)
        return img

    async def process_body_color_async(self, image, keypoints, print_fps=False):
        try:
            start_time = time.time()
            edge_pixels_tasks = [self.extract_edge_pixels_async(image, start, end, keypoints)
                                 for start, end in self.COCO_EDGES]
            edge_pixels = await asyncio.gather(*edge_pixels_tasks)
            color_signature_tasks = []
            for pixels in edge_pixels:
                if pixels is not None:
                    task = self.find_raw_dominant_color_async(pixels)
                else:
                    task = asyncio.sleep(0, result=np.full((3,), np.nan))
                color_signature_tasks.append(task)
            color_signature_list = await asyncio.gather(*color_signature_tasks)
            color_signature = np.array(color_signature_list)
            end_time = time.time()
            if print_fps:
                process_time = end_time - start_time
                fps = 1.0 / process_time if process_time > 0 else 0
                # logger.info(f"Body Color Processing FPS: {fps:.2f}")
            return color_signature
        except Exception as e:
            logger.error(f"Error in process_body_color: {str(e)}", exc_info=True)
            return None
        
def preprocess_color(body_color: np.ndarray, thigh_weight, torso_weight):
    """
    - body_color: ndarray shape (17, 3)  # 17 regions (COCO edges)
    - flatten → shape (51,)
    - NaN → thay bằng mean của các giá trị không NaN trong toàn vector
    - Áp dụng trọng số cho thigh (vùng 10, 12), torso (vùng 14, 15, 16)
    - normalize
    - Trả về: (vector đã chuẩn hóa, mask)
    """
    if body_color.shape != (17, 3):
        raise ValueError("body_color phải có shape (17, 3)")

    # Flatten and prepare mask
    flat = body_color.flatten().astype(np.float32)       # (51,)
    mask = (~np.isnan(flat)).astype(np.float32)          # (51,)

    # Compute global mean from non-NaN entries and fill NaNs
    mean_val = np.nanmean(flat)
    filled = np.where(np.isnan(flat), mean_val, flat)

    # Initialize weights
    weights = np.ones_like(filled, dtype=np.float32)

    # Identify regions by edge index
    # Thigh edges at indices 10 and 12
    for region in (10, 12):
        start, end = region * 3, region * 3 + 3
        weights[start:end] *= thigh_weight

    # Torso edges at indices 14, 15, 16
    for region in (14, 15, 16):
        start, end = region * 3, region * 3 + 3
        weights[start:end] *= torso_weight

    # Apply weights and normalize
    weighted = filled * weights

    return weighted, mask