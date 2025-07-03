import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import faiss
import time
from .logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Định nghĩa các cạnh COCO (không bao gồm đùi và torso)
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Kết nối khuôn mặt (4 edges)
    (5, 6),                               # Vai (1 edge)
    (5, 7), (7, 9),                       # Cánh tay trái (2 edges)
    (6, 8), (8, 10),                      # Cánh tay phải (2 edges)
    (11, 12),                             # Hông (1 edge)
    (13, 15),                             # Cẳng chân trái (1 edge)
    (14, 16)                              # Cẳng chân phải (1 edge)
]
# Tổng len(COCO_EDGES) = 12

# Các cạnh đùi (xử lý riêng để đặt thickness lớn hơn)
THIGH_EDGES = [
    (11, 13),  # Đùi trái
    (12, 14)   # Đùi phải
]

TORSO_EDGES = [
    (5, 11), (6, 12), (6, 11) 
]
# Tổng len(THIGH_EDGES) = 2

MIN_POINTS_FOR_CLUSTERING = 50


class PoseColorSignatureExtractor:
    def __init__(
        self,
        n_cluster: int = 3,
        dim: int = 3,
        n_iter: int = 50,
        verbose: bool = False,
        color_space: str = 'BGR',
        edge_thickness: int = 10,
        thigh_thickness: int = 20,
        torso_thickness: int = 25,
        max_workers: int = None
    ):
        self.n_cluster = n_cluster
        self.dim = dim
        self.n_iter = n_iter
        self.verbose = verbose
        self.color_space = color_space.upper()
        self.edge_thickness = edge_thickness
        self.thigh_thickness = thigh_thickness
        self.torso_thickness = torso_thickness

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _preprocess_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """
        Loại bỏ các pixel nằm ngoài percentiles 5-95 mỗi kênh màu.
        Nếu pixels trống, trả về nguyên bản.
        """
        if pixels.size == 0:
            return pixels

        mask = np.ones(pixels.shape[0], dtype=bool)
        for c in range(self.dim):
            lo, hi = np.percentile(pixels[:, c], [5, 95])
            mask &= (pixels[:, c] >= lo) & (pixels[:, c] <= hi)
        return pixels[mask]

    def _extract_edge_pixels(self, image: np.ndarray, start: int, end: int, keypoints: np.ndarray, thickness: int) -> np.ndarray:
        if max(start, end) >= len(keypoints):
            return None

        x1, y1 = map(int, keypoints[start])
        x2, y2 = map(int, keypoints[end])

        if (x1, y1) == (0, 0) or (x2, y2) == (0, 0):
            return None

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness)
        pixels = image[mask.astype(bool)]
        return pixels if pixels.size else None

    def _extract_torso_pixels(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Trích pixel vùng torso:
        - Nếu có ít nhất 2 keypoints hợp lệ, sẽ extract:
        - 2 điểm: nối đường thẳng với độ dày line thickness = 20
        - 3 điểm: nối đa giác 3 điểm và lấy pixel bên trong
        - 4 điểm: nối đa giác 4 điểm và lấy pixel bên trong
        - Nếu số điểm hợp lệ < 2 hoặc không xác định, trả về None.
        """
        # Các chỉ số vai-trái, hông-trái, hông-phải, vai-phải
        idxs = [5, 11, 12, 6]
        pts = []
        # Lọc các điểm hợp lệ
        for i in idxs:
            try:
                x, y = map(int, keypoints[i])
            except (IndexError, ValueError):
                continue
            if (x, y) == (0, 0):
                continue
            pts.append((x, y))

        # Nếu ít hơn 2 điểm hợp lệ
        if len(pts) < 2:
            return None

        # Tạo mặt nạ
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if len(pts) == 2:
            # Nối 2 điểm thành đường thẳng với độ dày 20
            cv2.line(mask, pts[0], pts[1], color=1, thickness=20)
        else:
            # Với 3 hoặc 4 điểm: tạo đa giác
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 1)

        # Lấy pixel từ image theo mask
        pixels = image[mask.astype(bool)]
        return pixels if pixels.size > 0 else None

    async def _to_async(self, func, *args):
        """
        Chạy hàm sync (func) trong ThreadPoolExecutor để không chặn event loop.
        Trả về kết quả của func(*args) dưới dạng asyncio.Future.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    def _compute_dominant(self, pixels: np.ndarray) -> np.ndarray:
        """
        Tính màu chủ đạo của một mảng pixel:
        - Nếu pixels là None hoặc trống, trả về None.
        - Nếu số pixel < max(n_cluster, MIN_POINTS_FOR_CLUSTERING), trả về trung bình (mean).
        - Ngược lại, tiến hành KMeans (FAISS) trên không gian 3 chiều (BGR hoặc LAB).
        - Trả về centroid của cluster có số pixel đông nhất, convert về BGR nếu cần.
        """
        if pixels is None or pixels.size == 0:
            return None

        filtered = self._preprocess_pixels(pixels)
        if filtered.shape[0] < max(self.n_cluster, MIN_POINTS_FOR_CLUSTERING):
            return filtered.mean(axis=0)

        data = filtered.astype(np.float32)
        if self.color_space == 'LAB':
            lab = cv2.cvtColor(
                np.clip(data, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
                cv2.COLOR_BGR2LAB
            ).reshape(-1, 3).astype(np.float32)
            data = lab

        kmeans = faiss.Kmeans(d=self.dim, k=self.n_cluster, niter=self.n_iter, verbose=self.verbose)
        kmeans.train(data)
        _, labels = kmeans.index.search(data, 1)
        counts = np.bincount(labels.ravel(), minlength=self.n_cluster)
        center = kmeans.centroids[counts.argmax()]

        if self.color_space == 'LAB':
            return cv2.cvtColor(
                np.clip(center, 0, 255).astype(np.uint8).reshape(1, 1, 3),
                cv2.COLOR_LAB2BGR
            )[0, 0]

        return center

    async def process_body_color_async(self, image: np.ndarray, keypoints: np.ndarray, print_fps: bool = False) -> np.ndarray:
        try:
            start_time = time.time()

            all_edges = COCO_EDGES + THIGH_EDGES + TORSO_EDGES

            # Tạo list độ dày tương ứng cho từng cạnh
            thickness_list = (
                [self.edge_thickness] * len(COCO_EDGES) +
                [self.thigh_thickness] * len(THIGH_EDGES) +
                [self.torso_thickness] * len(TORSO_EDGES)
            )

            edge_tasks = [
                self._to_async(self._extract_edge_pixels, image, s, e, keypoints, thickness)
                for (s, e), thickness in zip(all_edges, thickness_list)
            ]

            edge_pixels = await asyncio.gather(*edge_tasks)

            color_tasks = [
                self._to_async(self._compute_dominant, pix)
                for pix in edge_pixels
            ]

            colors = await asyncio.gather(*color_tasks)

            if print_fps:
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                # logger.info(f"Processed {len(colors)} regions in {elapsed:.3f}s ({fps:.2f} FPS)")

            result = np.array([
                c if c is not None else np.full(3, np.nan)
                for c in colors
            ], dtype=np.float32)

            return result

        except Exception as e:
            logger.exception("Error in process_body_color_async: %s", e)
            return np.full((len(COCO_EDGES) + len(THIGH_EDGES) + len(TORSO_EDGES), 3), np.nan, dtype=np.float32)

    # async def process_body_color_async(self, image: np.ndarray, keypoints: np.ndarray, print_fps: bool = False) -> np.ndarray:
    #     """
    #     Tính signature màu cho từng region:
    #     - Số region = len(COCO_EDGES) + len(THIGH_EDGES) + 1 (torso).
    #     - Trả về numpy.ndarray shape (15, 3):
    #         [colors_edge_0, …, colors_edge_13, colors_edge_14, color_torso]
    #     - Mỗi entry là vector BGR 3 chiều hoặc NaN nếu không tính được.
    #     """
    #     try:
    #         start_time = time.time()

    #         # 1. Kết hợp danh sách các edge cần xử lý
    #         all_edges = COCO_EDGES + THIGH_EDGES  # 12 + 2 = 14 edges

    #         # 2. Tạo tasks để extract pixel dọc mỗi edge
    #         edge_tasks = [
    #             self._to_async(self._extract_edge_pixels, image, s, e, keypoints)
    #             for (s, e) in all_edges
    #         ]

    #         # 3. Tạo task extract pixel vùng torso
    #         torso_task = self._to_async(self._extract_torso_pixels, image, keypoints)

    #         # 4. Chạy song song tất cả extract tasks
    #         edge_pixels = await asyncio.gather(*edge_tasks)      # list length = 14
    #         torso_pixels = await torso_task                        # 1 element

    #         # 5. Tạo tasks để tính màu chủ đạo cho từng list pixel
    #         color_tasks = [
    #             self._to_async(self._compute_dominant, pix)
    #             for pix in edge_pixels
    #         ]
    #         color_tasks.append(self._to_async(self._compute_dominant, torso_pixels))

    #         # 6. Chạy song song tasks tính dominant color
    #         colors = await asyncio.gather(*color_tasks)  # length = 15

    #         # 7. In thông tin FPS nếu cần
    #         if print_fps:
    #             elapsed = time.time() - start_time
    #             fps = 1.0 / elapsed if elapsed > 0 else 0
    #             logger.info(f"Processed {len(colors)} regions in {elapsed:.3f}s ({fps:.2f} FPS)")

    #         # 8. Chuyển None thành [nan, nan, nan]
    #         result = np.array([
    #             c if c is not None else np.full(3, np.nan)
    #             for c in colors
    #         ], dtype=np.float32)  # shape = (15, 3)

    #         return result

    #     except Exception as e:
    #         logger.exception("Error in process_body_color_async: %s", e)
    #         # Nếu có lỗi, trả về NaN cho toàn bộ 15 vùng
    #         return np.full((len(COCO_EDGES) + len(THIGH_EDGES) + 1, 3), np.nan, dtype=np.float32)

    def shutdown(self):
        """Tắt ThreadPoolExecutor khi không còn sử dụng nữa."""
        self.executor.shutdown(wait=False)


def preprocess_color(body_color: np.ndarray, thigh_weight, torso_weight):
    """
    - body_color: ndarray shape (17, 3)  # 17 regions (COCO edges)
    - flatten → shape (51,)
    - NaN → thay bằng mean của các giá trị không NaN trong toàn vector
    - Áp dụng trọng số cho thigh (vùng 12, 13), torso (vùng 14, 15, 16)
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

    # Vùng đùi (thigh edges): chỉ số là 12, 13
    for region in (12, 13):  # Đổi chỉ số vùng đùi
        start, end = region * 3, region * 3 + 3
        weights[start:end] *= thigh_weight

    # Vùng torso (torso edges): chỉ số là 14, 15, 16
    for region in (14, 15, 16):
        start, end = region * 3, region * 3 + 3
        weights[start:end] *= torso_weight

    # Áp dụng trọng số
    weighted = filled * weights

    # Normalize vector (chuẩn hóa vector)
    norm = np.linalg.norm(weighted)
    if norm > 0:
        weighted /= norm

    return weighted, mask
