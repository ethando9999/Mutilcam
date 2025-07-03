import numpy as np
from collections import OrderedDict, deque

from .kalman_filter import KalmanFilter, chi2inv95

from utils.logging_python_orangepi import get_logger

# Giả sử KalmanFilter đã được định nghĩa/import như bạn đưa:
# - state vector 8-d: [cx, cy, w, h, vx, vy, vw, vh]
# - measurement cũng [cx, cy, w, h]
# - phương thức predict, update, gating_distance như đã có.

logger = get_logger(__name__)

# Dùng chi-square 0.95 quantile bậc tự do 4:
CHI2INV95_4 = chi2inv95[4]


class TrackingManager:
    """
    Quản lý track state cho mỗi person_id, dùng KalmanFilter (8-d) + appearance.
    Mỗi track lưu:
      - mean (8-d), covariance (8x8)
      - last_frame_id, time_since_update
      - feature_history (deque) để tính moving average appearance
    """
    def __init__(self,
                 gating_threshold: float = CHI2INV95_4,
                 max_time_lost: int = 30,
                 proximity_thresh: float = 0.7,
                 appearance_thresh: float = 0.5,
                 feature_history_len: int = 5):
        """
        kalman_filter: instance của KalmanFilter đã định nghĩa.
        gating_threshold: squared Mahalanobis threshold (dùng full-state, do measurement dim=4).
        max_time_lost: số frame không update tối đa trước khi xóa track.
        proximity_thresh: ngưỡng IoU tối thiểu để candidate (ví dụ 0.7).
        appearance_thresh: ngưỡng cosine similarity tối thiểu để match appearance.
        feature_history_len: độ dài lịch sử feature để moving average.
        """
        self.kf = KalmanFilter()
        # tracks: OrderedDict giữ thứ tự insert (không bắt buộc nhưng duy trì stable order)
        # key: person_id, value: dict chứa mean, covariance, last_frame_id, time_since_update, feature_history
        self.tracks = OrderedDict()
        self.gating_threshold = gating_threshold
        self.max_time_lost = max_time_lost
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.feature_history_len = feature_history_len

    @staticmethod
    def bbox_tlbr_to_xywh(bbox):
        """
        Chuyển bbox [x1,y1,x2,y2] sang measurement [cx, cy, w, h].
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        return np.array([cx, cy, w, h], dtype=np.float32)

    @staticmethod
    def xywh_to_tlbr(xywh):
        """
        Chuyển [cx,cy,w,h] sang [x1,y1,x2,y2].
        """
        cx, cy, w, h = xywh
        x1 = cx - w/2.
        y1 = cy - h/2.
        x2 = cx + w/2.
        y2 = cy + h/2.
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def iou(b1, b2):
        """
        Tính IoU giữa hai bbox [x1,y1,x2,y2].
        """
        xa1, ya1, xa2, ya2 = b1
        xb1, yb1, xb2, yb2 = b2
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        iw = max(0., xi2 - xi1)
        ih = max(0., yi2 - yi1)
        inter = iw * ih
        area_a = max(0., xa2 - xa1) * max(0., ya2 - ya1)
        area_b = max(0., xb2 - xb1) * max(0., yb2 - yb1)
        union = area_a + area_b - inter
        if union <= 0.:
            return 0.0
        return inter / union

    def predict_all(self, current_frame_id):
        """
        Predict tất cả tracks đến current_frame_id. 
        Giả sử frame_id tăng đều 1 mỗi khung. Nếu nhảy nhiều frame, có thể lặp thêm.
        Ở đây đơn giản: nếu current_frame_id > last_frame_id, gọi predict 1 bước.
        """
        to_delete = []
        for pid, tr in list(self.tracks.items()):
            last = tr['last_frame_id']
            if current_frame_id > last:
                # gọi predict một bước
                mean, cov = tr['mean'], tr['covariance']
                new_mean, new_cov = self.kf.predict(mean.copy(), cov.copy())
                tr['mean'], tr['covariance'] = new_mean, new_cov
                tr['last_frame_id'] = current_frame_id
                tr['time_since_update'] += 1
            # sau predict, nếu time_since_update quá lớn, sẽ xóa ở _remove_stale_tracks
        self._remove_stale_tracks()

    def _remove_stale_tracks(self):
        """
        Xóa các track đã quá time_since_update > max_time_lost.
        """
        to_remove = [pid for pid, tr in self.tracks.items()
                     if tr['time_since_update'] > self.max_time_lost]
        for pid in to_remove:
            del self.tracks[pid]

    def add_track(self, person_id, bbox, frame_id, feature=None):
        """
        Tạo mới track cho person_id với bbox ([x1,y1,x2,y2]) và optional feature (normalized).
        Nếu track đã tồn tại, override lịch sử (reset).
        """
        measurement = self.bbox_tlbr_to_xywh(bbox)
        mean, cov = self.kf.initiate(measurement)
        hist = deque(maxlen=self.feature_history_len)
        if feature is not None:
            hist.append(feature.copy())
        self.tracks[person_id] = {
            'mean': mean,
            'covariance': cov,
            'last_frame_id': frame_id,
            'time_since_update': 0,
            'feature_history': hist
        }

    def update_track(self, person_id, bbox, frame_id, feature=None):
        """
        Cập nhật track đã tồn tại; nếu chưa có, gọi add_track.
        """
        if person_id not in self.tracks:
            self.add_track(person_id, bbox, frame_id, feature)
            return
        tr = self.tracks[person_id]
        # Predict nếu cần
        if frame_id > tr['last_frame_id']:
            mean, cov = tr['mean'], tr['covariance']
            new_mean, new_cov = self.kf.predict(mean.copy(), cov.copy())
            tr['mean'], tr['covariance'] = new_mean, new_cov
            tr['last_frame_id'] = frame_id
            tr['time_since_update'] += 1
        # Update measurement
        measurement = self.bbox_tlbr_to_xywh(bbox)
        mean, cov = tr['mean'], tr['covariance']
        new_mean, new_cov = self.kf.update(mean, cov, measurement)
        tr['mean'], tr['covariance'] = new_mean, new_cov
        tr['last_frame_id'] = frame_id
        tr['time_since_update'] = 0
        if feature is not None:
            tr['feature_history'].append(feature.copy())

    def match(self, bbox, frame_id, feature=None):
        """
        Thử match detection (bbox + optional feature) với các track hiện có:
        1) Predict all tracks đến frame_id.
        2) Spatial gating (Mahalanobis): giữ track có gating_distance <= gating_threshold.
        3) Với những candidate, tính IoU giữa predicted bbox và bbox; giữ nếu IoU >= proximity_thresh.
        4) Nếu feature có truyền vào:
             - Tính moving average feature của mỗi track từ feature_history.
             - Tính cosine similarity giữa avg_feat và feature.
             - Giữ nếu sim >= appearance_thresh.
        5) Chọn best:
             - Nếu có feature: chọn sim cao nhất (nếu tie, chọn IoU cao hơn).
             - Nếu không có feature: chọn IoU cao nhất.
        6) Nếu tìm được best_pid: gọi update_track(best_pid, ...), trả về best_pid.
        Nếu không, trả None.
        """
        if not self.tracks:
            logger.warning("KHÔNG có track nào tồn tại")
            return None

        # 1) predict_all như cũ…
        self.predict_all(frame_id)

        meas = self.bbox_tlbr_to_xywh(bbox).reshape(1,4)
        candidates = []
        for pid, tr in self.tracks.items():
            dist = self.kf.gating_distance(tr['mean'], tr['covariance'], meas,
                                            only_position=False, metric='maha')[0]
            # logger.warning(f"[DEBUG] pid={pid} mahalanobis={dist:.2f}")
            if dist <= self.gating_threshold:
                candidates.append(pid)
        if not candidates:
            # logger.info("→ no candidates after mahalanobis gating")
            return None

        iou_ok = []
        for pid in candidates:
            pred_tlbr = self.xywh_to_tlbr(self.tracks[pid]['mean'][:4])
            i = self.iou(pred_tlbr, np.array(bbox, dtype=np.float32))
            # logger.warning(f"[DEBUG] pid={pid} IoU={i:.2f}")
            if i >= self.proximity_thresh:
                iou_ok.append((pid, i))
        if not iou_ok:
            # logger.info("→ no candidates after IoU filter")
            return None

        if feature is not None:
            final = []
            for pid, iou_val in iou_ok:
                hist = self.tracks[pid]['feature_history']
                if not hist:
                    # logger.warning(f"[DEBUG] pid={pid} no appearance history")
                    continue
                feat_avg = np.mean(np.stack(hist), axis=0)
                sim = np.dot(feat_avg, feature) / (np.linalg.norm(feat_avg)*np.linalg.norm(feature))
                # logger.warning(f"[DEBUG] pid={pid} cos_sim={sim:.2f}")
                if sim >= self.appearance_thresh:
                    final.append((pid, iou_val, sim))
            if not final:
                # logger.info("→ no candidates after appearance filter")
                return None
        else:
            final = [(pid, iou_val, None) for pid, iou_val in iou_ok]

        # 5. Chọn best
        if feature is not None:
            # chọn sim cao nhất; nếu tie, chọn iou cao hơn
            best = max(final, key=lambda x: (x[2], x[1]))
            best_pid = best[0]
        else:
            # chọn iou cao nhất
            best = max(final, key=lambda x: x[1])
            best_pid = best[0]

        return best_pid

    def get_active_tracks(self):
        """
        Trả list các person_id hiện đang active (chưa stale).
        """
        return list(self.tracks.keys())
