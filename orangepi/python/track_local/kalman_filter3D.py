import numpy as np

class KalmanFilter3D:
    """
    Một lớp Kalman Filter 3D cho việc theo dõi đối tượng trong không gian 3D.

    Lớp này triển khai một bộ lọc Kalman với mô hình chuyển động vận tốc không đổi.
    Trạng thái (state) là một vector 6 chiều: [x, y, z, vx, vy, vz]
    (vị trí và vận tốc trên các trục x, y, z).
    Phép đo (measurement) là một vector 3 chiều: [x, y, z] (chỉ vị trí).
    """

    def __init__(self, dt: float, process_noise_std: float, measurement_noise_std: float):
        """
        Hàm khởi tạo cho KalmanFilter3D.

        Args:
            dt (float): Khoảng thời gian giữa các bước (time step).
            process_noise_std (float): Độ lệch chuẩn của nhiễu quá trình.
            measurement_noise_std (float): Độ lệch chuẩn của nhiễu đo lường.
        """
        # Khoảng thời gian
        self.dt = dt

        # Vector trạng thái [x, y, z, vx, vy, vz]
        # Khởi tạo là một vector cột 6x1
        self.x = np.zeros((6, 1))

        # Ma trận chuyển đổi trạng thái (A) - 6x6
        self.A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Ma trận đo lường (H) - 3x6
        # Chuyển đổi từ không gian trạng thái 6D sang không gian đo lường 3D
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Ma trận hiệp phương sai nhiễu quá trình (Q) - 6x6
        # Giả định nhiễu là độc lập và có cùng phương sai trên tất cả các chiều
        q = process_noise_std ** 2
        self.Q = np.eye(6) * q

        # Ma trận hiệp phương sai nhiễu đo lường (R) - 3x3
        r = measurement_noise_std ** 2
        self.R = np.eye(3) * r

        # Ma trận hiệp phương sai lỗi ước tính (P) - 6x6
        # Khởi tạo là ma trận đơn vị
        self.P = np.eye(6)

    def predict(self):
        """
        Dự đoán trạng thái tiếp theo của hệ thống.
        """
        # Dự đoán trạng thái: x_k = A * x_{k-1}
        self.x = self.A @ self.x

        # Dự đoán hiệp phương sai lỗi: P_k = A * P_{k-1} * A^T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: np.ndarray):
        """
        Cập nhật trạng thái với một phép đo mới.

        Args:
            z (np.ndarray): Vector đo lường 3 chiều [x, y, z].
        """
        # Đảm bảo z là một vector cột 3x1
        z = np.array(z).reshape(3, 1)

        # Sai số đo lường (innovation): y = z - H * x_k
        y = z - self.H @ self.x

        # Hiệp phương sai của sai số (innovation covariance): S = H * P_k * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain: K = P_k * H^T * S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Cập nhật trạng thái ước tính: x_k = x_k + K * y
        self.x = self.x + K @ y

        # Cập nhật hiệp phương sai lỗi: P_k = (I - K * H) * P_k
        I = np.eye(self.H.shape[1]) # Ma trận đơn vị 6x6
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """
        Lấy trạng thái hiện tại.

        Returns:
            np.ndarray: Vector trạng thái 1D [x, y, z, vx, vy, vz].
        """
        return self.x.flatten()

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # Tham số cho bộ lọc
    dt = 0.1  # Thời gian giữa các phép đo là 1 giây
    process_noise = 0.1  # Nhiễu trong mô hình chuyển động
    measurement_noise = 0.5  # Nhiễu trong cảm biến đo lường

    # Khởi tạo bộ lọc
    kf = KalmanFilter3D(dt, process_noise, measurement_noise)

    # Tạo một chuỗi các phép đo mô phỏng (ví dụ: một đối tượng di chuyển theo đường thẳng)
    measurements = [
        np.array([1, 1, 1]),
        np.array([2.1, 1.9, 3.2]),
        np.array([3.0, 3.1, 4.9]),
        np.array([3.8, 5.2, 6.8]),
        np.array([5.2, 6.9, 9.1]),
    ]

    print("Bắt đầu quá trình lọc Kalman...")
    print("-" * 30)

    for i, z in enumerate(measurements):
        # Dự đoán trạng thái tiếp theo
        kf.predict()

        # Cập nhật với phép đo mới
        kf.update(z)

        # Lấy trạng thái đã được lọc
        filtered_state = kf.get_state()

        print(f"Phép đo {i+1}: {z}")
        print(f"Trạng thái ước tính: [x, y, z] = [{filtered_state[0]:.2f}, {filtered_state[1]:.2f}, {filtered_state[2]:.2f}]")
        print(f"Vận tốc ước tính: [vx, vy, vz] = [{filtered_state[3]:.2f}, {filtered_state[4]:.2f}, {filtered_state[5]:.2f}]")
        print("-" * 30)
