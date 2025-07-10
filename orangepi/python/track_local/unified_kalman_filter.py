# file: python/tracking/unified_kalman_filter.py

import numpy as np
import scipy.linalg

# Bảng hằng số chi-square cho Mahalanobis gating
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

class UnifiedKalmanFilter:
    """
    Một bộ lọc Kalman hợp nhất để theo dõi trạng thái 2D và 3D của đối tượng.
    Không gian trạng thái (10 chiều): [cx, cy, w, h, z, v_cx, v_cy, v_w, v_h, v_z]
    Không gian đo lường (5 chiều):  [cx, cy, w, h, z]
    """

    def __init__(self, dt: float = 1.0):
        ndim, dt = 5, dt  # 5 chiều đo lường
        self._A = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._A[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)

        # Các trọng số nhiễu, có thể cần tinh chỉnh
        self._std_weight_position_2d = 1. / 20
        self._std_weight_velocity_2d = 1. / 160
        self._std_weight_depth = 1. / 30
        self._std_weight_velocity_depth = 1. / 200

    def initiate(self, measurement: np.ndarray):
        """Khởi tạo một track mới từ một phép đo 5D."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position_2d * measurement[2],
            2 * self._std_weight_position_2d * measurement[3],
            2 * self._std_weight_position_2d * measurement[2],
            2 * self._std_weight_position_2d * measurement[3],
            2 * self._std_weight_depth * measurement[4],
            10 * self._std_weight_velocity_2d * measurement[2],
            10 * self._std_weight_velocity_2d * measurement[3],
            10 * self._std_weight_velocity_2d * measurement[2],
            10 * self._std_weight_velocity_2d * measurement[3],
            10 * self._std_weight_velocity_depth * measurement[4]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Thực hiện bước dự đoán Kalman."""
        std_pos = [
            self._std_weight_position_2d * mean[2], self._std_weight_position_2d * mean[3],
            self._std_weight_position_2d * mean[2], self._std_weight_position_2d * mean[3],
            self._std_weight_depth * mean[4]
        ]
        std_vel = [
            self._std_weight_velocity_2d * mean[2], self._std_weight_velocity_2d * mean[3],
            self._std_weight_velocity_2d * mean[2], self._std_weight_velocity_2d * mean[3],
            self._std_weight_velocity_depth * mean[4]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._A @ mean
        covariance = self._A @ covariance @ self._A.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Chiếu trạng thái xuống không gian đo lường."""
        std = [
            self._std_weight_position_2d * mean[2], self._std_weight_position_2d * mean[3],
            self._std_weight_position_2d * mean[2], self._std_weight_position_2d * mean[3],
            self._std_weight_depth * mean[4]
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + innovation_cov
        return projected_mean, projected_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Thực hiện bước cập nhật Kalman."""
        projected_mean, projected_cov = self.project(mean, covariance)

        # <<< SỬA LỖI TOÁN HỌC: Sửa công thức tính Kalman Gain >>>
        # Công thức đúng: K = P @ H.T @ inv(S)
        # Tương đương với giải hệ phương trình: S @ K.T = H @ P
        # B = H @ P có shape (5, 10) x (10, 10) -> (5, 10)
        # S (projected_cov) có shape (5, 5). Giải S @ X = B cho X = K.T
        b = self._H @ covariance
        kalman_gain_T = scipy.linalg.solve(projected_cov, b, check_finite=False, assume_a='pos')
        kalman_gain = kalman_gain_T.T
        # <<< KẾT THÚC SỬA LỖI >>>
        
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray):
        """Tính khoảng cách Mahalanobis bình phương."""
        projected_mean, projected_cov = self.project(mean, covariance)
        
        measurements = np.atleast_2d(measurements)
        d = measurements - projected_mean
        
        # Sử dụng solve thay vì cholesky để tăng độ ổn định với các ma trận gần suy biến
        try:
            z = np.linalg.solve(np.linalg.cholesky(projected_cov), d.T)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        except np.linalg.LinAlgError:
            # Fallback nếu Cholesky thất bại
            inv_cov = np.linalg.inv(projected_cov)
            return np.sum(d * (inv_cov @ d.T).T, axis=1)