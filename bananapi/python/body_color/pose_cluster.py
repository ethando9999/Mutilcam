### File: pose_cluster_processor.py
import os
import cv2
import numpy as np
import faiss
from collections import defaultdict
import matplotlib.pyplot as plt

import time
from logging_python_bananapi import get_logger

logger = get_logger(__name__)

class PoseClusterProcessor:
    # Danh sách các cặp keypoint (edges) dựa trên COCO (Common Objects in Context) keypoint format.
    COCO_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
        (5, 11), (6, 12), (6, 11)
    ]

    def __init__(self, n_cluster=3, d=3, n_iter=100, verbose=False):
        """
        Khởi tạo PoseClusterProcessor.

        Args:
            n_cluster (int): Số lượng cụm màu (clusters) sử dụng trong KMeans.
            d (int): Số chiều dữ liệu màu (thường là 3 cho RGB).
            n_iter (int): Số vòng lặp cho thuật toán KMeans.
            verbose (bool): Bật/Tắt thông tin chi tiết khi huấn luyện KMeans.
        """
        self.n_cluster = n_cluster
        self.d = d
        self.n_iter = n_iter
        self.verbose = verbose
        self.feature_db = defaultdict(dict)  # Cơ sở dữ liệu lưu trữ đặc trưng và cụm màu.
        self.feature_cache = []  # Bộ nhớ tạm để lưu đặc trưng từ 1000 khung hình.

    def extract_edge_pixels(self, image, keypoints, thickness=15):
        """
        Trích xuất các pixel nằm trên các đường nối giữa keypoints.

        Args:
            image (numpy.ndarray): Ảnh đầu vào.
            keypoints (list): Danh sách tọa độ keypoints của cơ thể.
            thickness (int): Độ dày của đường nối khi vẽ.

        Returns:
            list: Danh sách các pixel tương ứng với mỗi cạnh.
        """
        edge_pixels = [None] * len(self.COCO_EDGES)
        for idx, (start, end) in enumerate(self.COCO_EDGES):
            # Kiểm tra keypoints hợp lệ
            if start >= len(keypoints) or end >= len(keypoints):
                continue

            pt1 = tuple(map(int, keypoints[start][:2]))  # Tọa độ điểm bắt đầu
            pt2 = tuple(map(int, keypoints[end][:2]))  # Tọa độ điểm kết thúc

            # Bỏ qua nếu keypoint không hợp lệ
            if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                continue

            # Tạo mask và trích xuất pixel trên đường nối
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.line(mask, pt1, pt2, 1, thickness)
            pixels = image[mask > 0]

            if len(pixels) > 0:
                edge_pixels[idx] = pixels  # Lưu pixel cho cạnh tương ứng

        return edge_pixels

    def find_dominant_color(self, pixels):
        """
        Xác định màu sắc chiếm ưu thế từ danh sách pixel sử dụng FAISS KMeans.

        Args:
            pixels (numpy.ndarray): Mảng các pixel.

        Returns:
            numpy.ndarray: Màu sắc chiếm ưu thế trong cụm.
        """
        if len(pixels) == 0:
            return None

        pixels = np.float32(pixels)  # Chuyển đổi pixel sang float32
        kmeans = faiss.Kmeans(self.d, self.n_cluster, niter=self.n_iter, verbose=self.verbose)
        kmeans.train(pixels)  # Huấn luyện KMeans trên pixel

        # Gán nhãn cho mỗi pixel vào cụm gần nhất
        _, labels = kmeans.index.search(pixels, 1)
        counts = np.bincount(labels.ravel(), minlength=self.n_cluster)  # Đếm số lượng pixel trong mỗi cụm
        dominant_cluster_idx = np.argmax(counts)  # Tìm cụm có nhiều pixel nhất
        dominant_color = kmeans.centroids[dominant_cluster_idx]  # Lấy màu từ cụm chiếm ưu thế

        return dominant_color

    def get_body_color_signature(self, edge_pixels):
        """
        Tạo chữ ký màu sắc cho cơ thể dựa trên các pixel của các cạnh.

        Args:
            edge_pixels (list): Danh sách pixel của các cạnh.

        Returns:
            list: Danh sách màu chiếm ưu thế cho mỗi cạnh.
        """
        color_signature = []
        for pixels in edge_pixels:
            if pixels is None or len(pixels) < self.n_cluster:
                color_signature.append(None)
                continue
            dominant_color = self.find_dominant_color(pixels)
            color_signature.append(dominant_color)  # Thêm màu chiếm ưu thế vào danh sách

        return color_signature

    # Hàm visualize nhiều chữ ký màu sắc
    def visualize_color_signatures(self, color_signatures):
        """
        Hiển thị chữ ký màu sắc của nhiều cơ thể dưới dạng biểu đồ.

        Args:
            color_signatures (list): Danh sách chữ ký màu sắc.
        """
        num_signatures = len(color_signatures)
        plt.figure(figsize=(12, num_signatures * 2))
        for i, color_signature in enumerate(color_signatures):
            for idx, color in enumerate(color_signature):
                if color is None:
                    color = [0, 0, 0]  # Màu đen cho các cạnh không có dữ liệu
                
                color = np.clip(color, 0, 255).astype(int)
                plt.subplot(num_signatures, len(color_signature), i * len(color_signature) + idx + 1)
                plt.imshow([[color / 255]])
                plt.axis('off')
                plt.title(f'Body {i+1}, Edge {idx}', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def process_body_color(self, image, keypoints, print_fps=False):
        """
        Xử lý và trích xuất chữ ký màu sắc từ ảnh và keypoints.
        
        Args:
            image (numpy.ndarray): Ảnh đầu vào
            keypoints (list): Danh sách các keypoints
            print_fps (bool): Có in FPS ra log hay không
            
        Returns:
            list: Chữ ký màu sắc của cơ thể
        """
        try:
            # Bắt đầu đo thời gian
            start_time = time.time()
            
            # Trích xuất pixel từ các cạnh
            edge_pixels = self.extract_edge_pixels(image, keypoints)
            
            # Lấy chữ ký màu sắc
            color_signature = self.get_body_color_signature(edge_pixels)
            
            # Tính và log FPS nếu được yêu cầu
            if print_fps:
                process_time = time.time() - start_time
                fps = 1.0 / process_time if process_time > 0 else 0
                logger.info(f"Body Color Processing FPS: {fps:.2f}")
            
            return color_signature
            
        except Exception as e:
            logger.error(f"Error in process_body_color: {str(e)}", exc_info=True)
            return None