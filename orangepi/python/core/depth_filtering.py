import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class DepthProcessor:
    def __init__(self, file_path, sparse_threshold=0.2, bilateral_d=15, sigma_color=0.2, sigma_space=15):
        self.file_path = file_path
        self.sparse_threshold = sparse_threshold
        self.bilateral_d = bilateral_d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

        
        self.depth = self.load_depth()
        self.normalized_depth = self.normalize_depth(self.depth)
        self.sparse_depth = self.create_sparse_depth(self.normalized_depth)

        self.filtered_depth = None
        self.fps = None

    def load_depth(self):
        return np.load(self.file_path).astype(np.float32)

    def normalize_depth(self, depth):
        depth_min, depth_max = depth.min(), depth.max()
        return (depth - depth_min) / (depth_max - depth_min + 1e-8)

    def create_sparse_depth(self, normalized_depth):
        sparse = normalized_depth.copy()
        sparse[sparse < self.sparse_threshold] = 0
        return sparse

    def apply_bilateral_filter(self):
        start_time = time.time()
        self.filtered_depth = cv2.bilateralFilter(
            src=self.sparse_depth,
            d=self.bilateral_d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )
        end_time = time.time()
        self.fps = 1.0 / (end_time - start_time)

    def show_results(self):
        if self.filtered_depth is None:
            raise ValueError("Bilateral filter chưa được áp dụng. Gọi `apply_bilateral_filter()` trước.")

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.title("Sparse Depth")
        plt.imshow(self.sparse_depth, cmap='plasma')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Completed (Bilateral)")
        plt.imshow(self.filtered_depth, cmap='plasma')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(self.normalized_depth, cmap='plasma')
        plt.colorbar()

        plt.suptitle(f"Bilateral Filter FPS: {self.fps:.2f}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    processor = DepthProcessor("D:/test1/depth_003.npy")
    processor.apply_bilateral_filter()
    processor.show_results()
