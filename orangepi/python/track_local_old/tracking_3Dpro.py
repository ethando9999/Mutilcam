import numpy as np
from collections import OrderedDict, deque
import scipy.linalg
import logging

# --- ĐỊNH NGHĨA LỚP KALMANFILTER3D ---

chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}

class KalmanFilter3D:
    def __init__(self, dt: float = 1.0, process_noise_std: float = 1.0, measurement_noise_std: float = 1.0):
        self.dt = dt
        self._A = np.array([
            [1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ])
        self._H = np.array([
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]
        ])
        q = process_noise_std ** 2
        self._Q = np.eye(6) * q
        r = measurement_noise_std ** 2
        self._R = np.eye(3) * r

    def initiate(self, measurement: np.ndarray):
        mean = np.zeros((6, 1))
        mean[:3] = np.array(measurement).reshape((3, 1))
        std = np.array([self._R[0, 0]**0.5, self._R[1, 1]**0.5, self._R[2, 2]**0.5, 10., 10., 10.])
        covariance = np.diag(std**2)
        return mean, covariance

    def _predict_stateless(self, mean: np.ndarray, covariance: np.ndarray):
        mean = self._A @ mean
        covariance = self._A @ covariance @ self._A.T + self._Q
        return mean, covariance

    def _update_stateless(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)
        measurement = np.array(measurement).reshape((3, 1))
        kalman_gain = scipy.linalg.cho_solve(
            (scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)),
            (covariance @ self._H.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + self._R
        return projected_mean, projected_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)
        measurements = np.atleast_2d(measurements)
        d = measurements - projected_mean.flatten()
        try:
            cholesky_factor = np.linalg.cholesky(projected_cov)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            squared_maha = np.array([np.inf] * measurements.shape[0])
        return squared_maha

# --- ĐỊNH NGHĨA LỚP TRACKINGMANAGER3D ---

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TrackingManager3D:
    def __init__(self, dt: float = 1.0, process_noise_std: float = 0.1, measurement_noise_std: float = 0.5,
                 gating_threshold: float = chi2inv95[3], max_time_lost: int = 30,
                 appearance_thresh: float = 0.5, feature_history_len: int = 10):
        self.kf = KalmanFilter3D(dt, process_noise_std, measurement_noise_std)
        self.tracks = OrderedDict()
        self.gating_threshold = gating_threshold
        self.max_time_lost = max_time_lost
        self.appearance_thresh = appearance_thresh
        self.feature_history_len = feature_history_len
        self._next_id = 1

    def predict_all(self):
        for tr in self.tracks.values():
            tr['mean'], tr['covariance'] = self.kf._predict_stateless(tr['mean'], tr['covariance'])
            tr['time_since_update'] += 1

    def _remove_stale_tracks(self):
        to_remove = [pid for pid, tr in self.tracks.items() if tr['time_since_update'] > self.max_time_lost]
        for pid in to_remove:
            logger.info(f"Xóa track ID {pid} do đã mất tích quá lâu (time_since_update={self.tracks[pid]['time_since_update']}).")
            del self.tracks[pid]

    def update(self, detections):
        # 1. Dự đoán tất cả các track hiện có
        self.predict_all()

        # [FIXED] Xử lý trường hợp không có detection
        if not detections:
            self._remove_stale_tracks()
            return

        # 2. Xây dựng ma trận chi phí (cost matrix)
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.full((num_tracks, num_dets), np.inf)
        track_pids = list(self.tracks.keys())

        if num_tracks > 0:
            for i, pid in enumerate(track_pids):
                tr = self.tracks[pid]
                det_points = np.array([d['point'] for d in detections])
                distances = self.kf.gating_distance(tr['mean'], tr['covariance'], det_points)
                
                for j, dist in enumerate(distances):
                    if dist > self.gating_threshold:
                        continue
                    
                    cost = dist
                    if 'feature' in detections[j] and tr['feature_history']:
                        feat_avg = np.mean(np.stack(tr['feature_history']), axis=0)
                        sim = np.dot(feat_avg, detections[j]['feature'])
                        if sim < self.appearance_thresh:
                            continue
                        cost = 1.0 - sim
                    
                    cost_matrix[i, j] = cost

        # 3. Gán cặp (matching)
        from scipy.optimize import linear_sum_assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for ti, di in zip(track_indices, det_indices):
            if cost_matrix[ti, di] != np.inf:
                matches.append((track_pids[ti], di))

        # 4. Cập nhật các track đã khớp
        matched_pids = {m[0] for m in matches}
        matched_dets_indices = {m[1] for m in matches}

        for pid, di in matches:
            tr = self.tracks[pid]
            det = detections[di]
            tr['mean'], tr['covariance'] = self.kf._update_stateless(tr['mean'], tr['covariance'], det['point'])
            if 'feature' in det:
                tr['feature_history'].append(det['feature'])
            tr['time_since_update'] = 0

        # 5. Tạo track mới cho các detection không khớp
        unmatched_dets_indices = set(range(num_dets)) - matched_dets_indices
        for di in unmatched_dets_indices:
            det = detections[di]
            mean, cov = self.kf.initiate(det['point'])
            hist = deque(maxlen=self.feature_history_len)
            if 'feature' in det:
                hist.append(det['feature'])
            
            self.tracks[self._next_id] = {
                'mean': mean, 'covariance': cov, 'time_since_update': 0, 'feature_history': hist
            }
            logger.info(f"Tạo track mới ID {self._next_id} từ detection tại {det['point']}.")
            self._next_id += 1

        # 6. Xóa các track cũ
        self._remove_stale_tracks()

# --- VÍ DỤ SỬ DỤNG ĐÃ SỬA LỖI ---
if __name__ == '__main__':
    manager = TrackingManager3D(max_time_lost=5, appearance_thresh=0.8)
    
    print("--- FRAME 0 ---")
    initial_detections = [{'point': np.array([10, 10, 10]), 'feature': np.array([0.9, 0.1]) / np.linalg.norm([0.9, 0.1])}]
    manager.update(initial_detections)
    print(f"Các track đang hoạt động: {list(manager.tracks.keys())}")
    print(f"Vị trí ước tính của Track 1: {np.round(manager.tracks[1]['mean'][:3].flatten(), 2)}")

    print("\n--- FRAME 1 ---")
    detections_f1 = [
        {'point': np.array([10.2, 10.3, 10.1]), 'feature': np.array([0.9, 0.1]) / np.linalg.norm([0.9, 0.1])},
        {'point': np.array([50, 50, 50]),       'feature': np.array([0.1, 0.9]) / np.linalg.norm([0.1, 0.9])},
    ]
    manager.update(detections_f1)
    print(f"Các track đang hoạt động: {list(manager.tracks.keys())}")
    print(f"Vị trí ước tính của Track 1: {np.round(manager.tracks[1]['mean'][:3].flatten(), 2)}")
    print(f"Vị trí ước tính của Track 2: {np.round(manager.tracks[2]['mean'][:3].flatten(), 2)}")

    for i in range(2, 8):
        print(f"\n--- FRAME {i} ---")
        manager.update([])
        print(f"Các track đang hoạt động: {list(manager.tracks.keys())}")
        if 1 in manager.tracks:
            print(f"  -> Track 1 time_since_update: {manager.tracks[1]['time_since_update']}")
        if 2 in manager.tracks:
            print(f"  -> Track 2 time_since_update: {manager.tracks[2]['time_since_update']}")
            
    print("\nHoàn thành ví dụ.")
