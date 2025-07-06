import numpy as np
import scipy.linalg

# Bảng định lượng 0.95 của phân phối chi-square, dùng làm ngưỡng cho Mahalanobis gating.
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147, # Sẽ dùng giá trị này cho không gian đo lường 3D
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter3D:
    """
    Một lớp Kalman Filter 3D được nâng cấp, lấy cảm hứng từ các triển khai chuyên nghiệp
    cho bài toán theo dõi đa đối tượng.

    Các cải tiến chính:
    1.  **Thiết kế lai (Hybrid Design):** Cung cấp cả API stateful (giống bản gốc)
        và các phương thức stateless (giống code 2) để tăng tính linh hoạt.
    2.  **Độ ổn định số học:** Sử dụng phân rã Cholesky (scipy.linalg.cho_solve)
        thay vì tính ma trận nghịch đảo trực tiếp, giúp tăng độ ổn định.
    3.  **Hỗ trợ Data Association:** Thêm phương thức `project` và `gating_distance`
        để tính khoảng cách Mahalanobis, rất quan trọng cho việc theo dõi đa đối tượng.

    Trạng thái (state) 6D: [x, y, z, vx, vy, vz]
    Phép đo (measurement) 3D: [x, y, z]
    """

    def __init__(self, dt: float = 1.0, process_noise_std: float = 1.0, measurement_noise_std: float = 1.0):
        """
        Hàm khởi tạo cho KalmanFilter3D.

        Args:
            dt (float): Khoảng thời gian giữa các bước (time step).
            process_noise_std (float): Độ lệch chuẩn của nhiễu quá trình.
            measurement_noise_std (float): Độ lệch chuẩn của nhiễu đo lường.
        """
        self.dt = dt

        # --- Các ma trận mô hình (Model Matrices) ---
        # Ma trận chuyển đổi trạng thái (A)
        self._A = np.array([
            [1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ])
        # Ma trận đo lường (H)
        self._H = np.array([
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]
        ])
        # Ma trận hiệp phương sai nhiễu quá trình (Q)
        q = process_noise_std ** 2
        self._Q = np.eye(6) * q
        # Ma trận hiệp phương sai nhiễu đo lường (R)
        r = measurement_noise_std ** 2
        self._R = np.eye(3) * r

        # --- Trạng thái nội tại (Internal State) cho API stateful ---
        self._x = np.zeros((6, 1))
        self._P = np.eye(6)

    # --- API Stateful (Giống bản gốc, tiện lợi cho việc theo dõi 1 đối tượng) ---

    def predict(self):
        """
        [Stateful] Dự đoán trạng thái tiếp theo và cập nhật trạng thái nội tại.
        """
        self._x, self._P = self._predict_stateless(self._x, self._P)

    def update(self, z: np.ndarray):
        """
        [Stateful] Cập nhật trạng thái nội tại với một phép đo mới.
        """
        self._x, self._P = self._update_stateless(self._x, self._P, z)

    def get_state(self) -> np.ndarray:
        """
        [Stateful] Lấy trạng thái nội tại hiện tại.
        """
        return self._x.flatten()

    def get_covariance(self) -> np.ndarray:
        """
        [Stateful] Lấy ma trận hiệp phương sai lỗi nội tại.
        """
        return self._P

    # --- API Stateless (Linh hoạt cho theo dõi đa đối tượng) ---

    def initiate(self, measurement: np.ndarray):
        """
        [Stateless] Tạo một track mới từ một phép đo.

        Args:
            measurement (np.ndarray): Vector đo lường 3 chiều [x, y, z].

        Returns:
            (np.ndarray, np.ndarray): Mean (6x1) và Covariance (6x6) của track mới.
        """
        mean = np.zeros((6, 1))
        mean[:3] = np.array(measurement).reshape((3, 1))
        
        # Khởi tạo hiệp phương sai với độ không chắc chắn cao cho vận tốc
        std = np.array([
            self._R[0, 0]**0.5, self._R[1, 1]**0.5, self._R[2, 2]**0.5,
            10., 10., 10. # Độ không chắc chắn lớn cho vận tốc ban đầu
        ])
        covariance = np.diag(std**2)
        return mean, covariance

    def _predict_stateless(self, mean: np.ndarray, covariance: np.ndarray):
        """
        [Stateless Core] Thực hiện bước dự đoán Kalman.
        """
        mean = self._A @ mean
        covariance = self._A @ covariance @ self._A.T + self._Q
        return mean, covariance

    def _update_stateless(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """
        [Stateless Core] Thực hiện bước cập nhật Kalman.
        """
        # Chiếu xuống không gian đo lường
        projected_mean, projected_cov = self.project(mean, covariance)

        # Đảm bảo measurement là vector cột
        measurement = np.array(measurement).reshape((3, 1))
        
        # Tính Kalman Gain một cách ổn định
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), covariance @ self._H.T, check_finite=False).T

        # Tính sai số
        innovation = measurement - projected_mean

        # Cập nhật mean và covariance
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """
        [Stateless Utility] Chiếu trạng thái và hiệp phương sai xuống không gian đo lường.
        """
        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + self._R
        return projected_mean, projected_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray):
        """
        [Stateless Utility] Tính khoảng cách Mahalanobis bình phương giữa phân phối
        trạng thái và các phép đo.

        Args:
            mean (np.ndarray): Vector trung bình của trạng thái (6x1).
            covariance (np.ndarray): Ma trận hiệp phương sai của trạng thái (6x6).
            measurements (np.ndarray): Mảng các phép đo kích thước (N, 3).

        Returns:
            np.ndarray: Một mảng 1D chứa khoảng cách Mahalanobis bình phương cho mỗi phép đo.
        """
        # Chiếu xuống không gian đo lường
        projected_mean, projected_cov = self.project(mean, covariance)

        # Đảm bảo measurements có shape (N, 3)
        measurements = np.atleast_2d(measurements)
        if measurements.shape[1] != 3:
            raise ValueError("Measurements must have shape (N, 3)")

        # Tính khoảng cách Mahalanobis
        d = measurements - projected_mean.flatten()
        cholesky_factor = np.linalg.cholesky(projected_cov)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # --- 1. Ví dụ sử dụng API Stateful (giống bản gốc) ---
    print("--- VÍ DỤ 1: SỬ DỤNG API STATEFUL ---")
    kf_stateful = KalmanFilter3D(dt=1.0, process_noise_std=0.1, measurement_noise_std=1.0)
    measurements_1 = [
        np.array([1, 1, 1]), np.array([2.1, 1.9, 3.2]), np.array([3.0, 3.1, 4.9])
    ]
    for i, z in enumerate(measurements_1):
        kf_stateful.predict()
        kf_stateful.update(z)
        state = kf_stateful.get_state()
        print(f"Phép đo {i+1}: {z}")
        print(f"  -> Trạng thái ước tính: [x,y,z] = [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
        print(f"  -> Vận tốc ước tính: [vx,vy,vz] = [{state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f}]")
    print("\n" + "="*50 + "\n")

    # --- 2. Ví dụ sử dụng API Stateless và Gating cho theo dõi đa đối tượng ---
    print("--- VÍ DỤ 2: SỬ DỤNG API STATELESS VÀ GATING ---")
    
    # Giả sử chúng ta đang theo dõi 2 đối tượng (track_1, track_2)
    # và nhận được 3 phép đo mới (detections).
    kf_stateless = KalmanFilter3D(dt=1.0, process_noise_std=0.1, measurement_noise_std=0.5)

    # Khởi tạo 2 tracks từ các phép đo trước đó
    mean1, cov1 = kf_stateless.initiate([10, 10, 10])
    mean2, cov2 = kf_stateless.initiate([30, 30, 30])
    
    # Thực hiện bước dự đoán cho cả 2 tracks
    predicted_mean1, predicted_cov1 = kf_stateless._predict_stateless(mean1, cov1)
    predicted_mean2, predicted_cov2 = kf_stateless._predict_stateless(mean2, cov2)

    print(f"Track 1 dự đoán ở vị trí: [{predicted_mean1[0][0]:.2f}, {predicted_mean1[1][0]:.2f}, {predicted_mean1[2][0]:.2f}]")
    print(f"Track 2 dự đoán ở vị trí: [{predicted_mean2[0][0]:.2f}, {predicted_mean2[1][0]:.2f}, {predicted_mean2[2][0]:.2f}]")
    
    # Các phép đo mới nhận được
    new_detections = np.array([
        [10.8, 10.9, 11.1], # Gần với track 1
        [50.0, 50.0, 50.0], # Xa, có thể là nhiễu hoặc đối tượng mới
        [30.5, 30.6, 30.4]  # Gần với track 2
    ])
    print(f"\nCác phép đo mới:\n{new_detections}")

    # Tính khoảng cách gating cho từng track với tất cả các phép đo
    dist1 = kf_stateless.gating_distance(predicted_mean1, predicted_cov1, new_detections)
    dist2 = kf_stateless.gating_distance(predicted_mean2, predicted_cov2, new_detections)

    print(f"\nKhoảng cách Mahalanobis^2 từ Track 1 đến các phép đo: {np.round(dist1, 2)}")
    print(f"Khoảng cách Mahalanobis^2 từ Track 2 đến các phép đo: {np.round(dist2, 2)}")

    # Sử dụng ngưỡng chi-square để quyết định việc ghép cặp
    gating_threshold = chi2inv95[3] # 3 bậc tự do cho không gian đo lường [x, y, z]
    print(f"\nNgưỡng Gating (chi2inv95[3]): {gating_threshold:.4f}")

    matches1 = dist1 < gating_threshold
    matches2 = dist2 < gating_threshold

    print(f"Phép đo nào khớp với Track 1? {matches1} -> Phép đo có index {np.where(matches1)[0]}")
    print(f"Phép đo nào khớp với Track 2? {matches2} -> Phép đo có index {np.where(matches2)[0]}")
