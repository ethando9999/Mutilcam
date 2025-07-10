# file: python/tracking/track_3d_pro.py

import numpy as np
from collections import OrderedDict

# Import các lớp đã được tối ưu và bộ lọc 1D
# Đảm bảo các đường dẫn import này là chính xác trong cấu trúc dự án của bạn
from .unified_kalman_filter import UnifiedKalmanFilter, chi2inv95
from core.stereo_projector_final import StereoProjectorFinal
from utils.kalman_filter import SimpleKalmanFilter
import logging

logger = logging.getLogger(__name__)

class Track3DPro:
    # ... (Toàn bộ nội dung của lớp Track3DPro giữ nguyên không đổi) ...
    """
    Quản lý việc theo dõi đa đối tượng bằng cách sử dụng bộ lọc Kalman hợp nhất
    và phép chiếu stereo mạnh mẽ.
    Tích hợp bộ lọc 1D để làm mịn giá trị chiều cao cho mỗi track.
    """
    def __init__(self, calib_file_path: str, max_time_lost: int = 30):
        # Giả định StereoProjector được khởi tạo đúng cách
        # Nếu StereoProjector của bạn có các tham số khác, hãy cập nhật ở đây
        self.stereo_projector = StereoProjectorFinal(calib_file_path=calib_file_path)
        self.kf = UnifiedKalmanFilter()
        
        self.tracks = OrderedDict()
        self.next_track_id = 0
        self.max_time_lost = max_time_lost
        self.gating_thresh = chi2inv95[5]  # Ngưỡng cho không gian đo lường 5D

    @staticmethod
    def bbox_to_xywh(bbox: list | np.ndarray) -> np.ndarray:
        """Chuyển đổi [x1, y1, x2, y2] sang [cx, cy, w, h]."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w / 2., y1 + h / 2., w, h])

    def _get_measurement(self, detection: dict) -> np.ndarray | None:
        """Từ một detection, tạo ra một phép đo 5D [cx, cy, w, h, z]."""
        bbox = detection.get('bbox')
        tof_depth_map = detection.get('tof_depth_map')
        if bbox is None or tof_depth_map is None:
            return None

        # Sử dụng get_robust_distance trả về tuple (distance, status)
        depth, status = self.stereo_projector.get_robust_distance(bbox, tof_depth_map)
        if status != "OK" or depth is None:
            return None
        
        xywh = self.bbox_to_xywh(bbox)
        return np.array([xywh[0], xywh[1], xywh[2], xywh[3], depth])

    def predict_all(self):
        """Dự đoán trạng thái tiếp theo cho tất cả các track."""
        for tr in self.tracks.values():
            tr['mean'], tr['covariance'] = self.kf.predict(tr['mean'], tr['covariance'])
            tr['time_since_update'] += 1

    def match(self, detections: list[dict]) -> tuple[list, list]:
        """Thực hiện việc ghép cặp giữa các track và các detection mới."""
        self.predict_all()
        
        valid_detections_map = {i: self._get_measurement(det) for i, det in enumerate(detections)}
        valid_measurements = [m for m in valid_detections_map.values() if m is not None]

        if not self.tracks or not valid_measurements:
            unmatched_detections = list(range(len(detections)))
            return [], unmatched_detections

        cost_matrix = np.full((len(self.tracks), len(valid_measurements)), np.inf)
        track_ids = list(self.tracks.keys())
        measurement_indices = [i for i, m in valid_detections_map.items() if m is not None]
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            distances = self.kf.gating_distance(track['mean'], track['covariance'], np.array(valid_measurements))
            cost_matrix[i, :] = distances

        matches = []
        used_detections = set()
        row_ind, col_ind = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.gating_thresh:
                break
            
            track_id = track_ids[r]
            detection_idx = measurement_indices[c]
            
            if track_id not in [m[0] for m in matches] and detection_idx not in used_detections:
                matches.append((track_id, detection_idx))
                used_detections.add(detection_idx)

        unmatched_tracks = set(self.tracks.keys()) - {m[0] for m in matches}
        unmatched_detections = set(range(len(detections))) - used_detections
        
        for track_id, det_idx in matches:
            measurement = valid_detections_map[det_idx]
            self.update_main_state(track_id, measurement)
        
        self._remove_stale_tracks(unmatched_tracks)
        
        return matches, list(unmatched_detections)

    def update_main_state(self, track_id: int, measurement: np.ndarray):
        """Cập nhật trạng thái Kalman hợp nhất (vị trí 2D & 3D)."""
        track = self.tracks[track_id]
        track['mean'], track['covariance'] = self.kf.update(track['mean'], track['covariance'], measurement)
        track['time_since_update'] = 0

    def register_new_track(self, detection: dict) -> int | None:
        """Đăng ký một track mới và khởi tạo bộ lọc chiều cao cho nó."""
        measurement = self._get_measurement(detection)
        if measurement is None:
            return None

        mean, covariance = self.kf.initiate(measurement)
        new_track_id = self.next_track_id
        
        self.tracks[new_track_id] = {
            'mean': mean, 'covariance': covariance, 'time_since_update': 0,
            'height_smoother': SimpleKalmanFilter(process_variance=1e-3, measurement_variance=2e-2),
            'last_smoothed_height': None
        }
        logger.info(f"✨ Đã đăng ký track mới ID: {new_track_id}")
        self.next_track_id += 1
        return new_track_id

    def update_height(self, track_id: int, raw_height: float) -> float | None:
        """Cập nhật chiều cao cho một track bằng bộ lọc 1D."""
        track = self.tracks.get(track_id)
        if track is None:
            logger.warning(f"Không tìm thấy track ID {track_id} để cập nhật chiều cao.")
            return None
        
        smoother = track['height_smoother']
        smoothed_height = smoother.update(raw_height)
        track['last_smoothed_height'] = smoothed_height
        
        return smoothed_height

    def _remove_stale_tracks(self, unmatched_track_ids: set):
        """Xóa các track đã mất dấu quá lâu."""
        to_remove = [tid for tid in unmatched_track_ids if self.tracks[tid]['time_since_update'] > self.max_time_lost]
        for tid in to_remove:
            logger.info(f"🗑️ Xóa track đã mất dấu ID: {tid}")
            if tid in self.tracks:
                del self.tracks[tid]