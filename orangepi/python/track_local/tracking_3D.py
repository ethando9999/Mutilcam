import numpy as np
from collections import OrderedDict, deque

# Import lớp KalmanFilter3D đã được nâng cấp và hằng số chi2inv95
from .kalman_filter_3D_pro import KalmanFilter3D, chi2inv95
from .kalman_filter_2D import KalmanFilter2D

# Giả sử logger đã được thiết lập
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



logger = logging.getLogger(__name__)

class TrackingManager3D:
    """
    Quản lý trạng thái track bằng cách kết hợp bộ lọc Kalman 2D (cho bbox) và
    bộ lọc Kalman 3D (cho điểm 3D) cho mỗi ID đối tượng.
    Phiên bản này có thể xử lý các trường hợp `point3d` là `None`.
    """
    def __init__(self,
                 max_time_lost: int = 30,
                 proximity_thresh: float = 0.5,
                 gating_thresh_2d: float = chi2inv95[4],
                 gating_thresh_3d: float = chi2inv95[3]):
        """
        Args:
            max_time_lost (int): Số khung hình tối đa một track có thể bị mất trước khi bị xóa.
            proximity_thresh (float): Ngưỡng IoU tối thiểu để kết hợp 2D.
            gating_thresh_2d (float): Ngưỡng Mahalanobis cho bộ lọc 2D.
            gating_thresh_3d (float): Ngưỡng Mahalanobis cho bộ lọc 3D.
        """
        self.kf2d = KalmanFilter2D()
        self.kf3d = KalmanFilter3D()
        self.tracks = OrderedDict()
        self.max_time_lost = max_time_lost
        self.proximity_thresh = proximity_thresh
        self.gating_thresh_2d = gating_thresh_2d
        self.gating_thresh_3d = gating_thresh_3d

    @staticmethod
    def bbox_tlbr_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w / 2., y1 + h / 2., w, h], dtype=np.float32)

    @staticmethod
    def xywh_to_tlbr(xywh):
        cx, cy, w, h = xywh
        return np.array([cx - w/2., cy - h/2., cx + w/2., cy + h/2.], dtype=np.float32)

    @staticmethod
    def iou(b1, b2):
        xi1, yi1 = np.maximum(b1[:2], b2[:2])
        xi2, yi2 = np.minimum(b1[2:], b2[2:])
        inter_area = np.prod(np.maximum(0., xi2 - xi1))
        b1_area = np.prod(b1[2:] - b1[:2])
        b2_area = np.prod(b2[2:] - b2[:2])
        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def predict_all(self, current_frame_id: int):
        """Dự đoán tất cả các track đến khung hình hiện tại và xóa các track cũ."""
        for tr in self.tracks.values():
            if current_frame_id > tr['last_frame_id']:
                # Luôn dự đoán trạng thái 2D
                tr['kf2d']['mean'], tr['kf2d']['covariance'] = self.kf2d.predict(tr['kf2d']['mean'], tr['kf2d']['covariance'])
                
                # Chỉ dự đoán trạng thái 3D nếu nó tồn tại
                if 'kf3d' in tr:
                    tr['kf3d']['mean'], tr['kf3d']['covariance'] = self.kf3d.predict(tr['kf3d']['mean'], tr['kf3d']['covariance'])
                
                tr['time_since_update'] += (current_frame_id - tr['last_frame_id'])
                tr['last_frame_id'] = current_frame_id
        self._remove_stale_tracks()

    def _remove_stale_tracks(self):
        to_remove = [pid for pid, tr in self.tracks.items() if tr['time_since_update'] > self.max_time_lost]
        for pid in to_remove:
            logger.info(f"Removing stale track ID: {pid}")
            del self.tracks[pid]

    def add_track(self, object_id, bbox, point3d, frame_id: int):
        """Khởi tạo và thêm một track mới. Trạng thái 3D là tùy chọn."""
        # Khởi tạo trạng thái 2D (bắt buộc)
        meas2d = self.bbox_tlbr_to_xywh(bbox)
        mean2d, cov2d = self.kf2d.initiate(meas2d)
        
        track_data = {
            'kf2d': {'mean': mean2d, 'covariance': cov2d},
            'last_frame_id': frame_id,
            'time_since_update': 0,
        }

        # Khởi tạo trạng thái 3D (tùy chọn)
        if point3d is not None:
            mean3d, cov3d = self.kf3d.initiate(np.array(point3d))
            track_data['kf3d'] = {'mean': mean3d, 'covariance': cov3d}
            logger.info(f"Adding new track {object_id} with 2D and 3D state.")
        else:
            logger.info(f"Adding new track {object_id} with only 2D state.")
            
        self.tracks[object_id] = track_data

    def update_track(self, object_id, bbox, point3d, frame_id: int):
        """Cập nhật một track hiện có với các đo lường mới."""
        if object_id not in self.tracks:
            self.add_track(object_id, bbox, point3d, frame_id)
            return

        tr = self.tracks[object_id]
        if frame_id > tr['last_frame_id']:
             self.predict_all(frame_id)

        # Cập nhật trạng thái 2D
        meas2d = self.bbox_tlbr_to_xywh(bbox)
        tr['kf2d']['mean'], tr['kf2d']['covariance'] = self.kf2d.update(tr['kf2d']['mean'], tr['kf2d']['covariance'], meas2d)
        
        # Cập nhật hoặc khởi tạo trạng thái 3D nếu có point3d
        if point3d is not None:
            if 'kf3d' in tr:
                # Cập nhật trạng thái 3D hiện có
                tr['kf3d']['mean'], tr['kf3d']['covariance'] = self.kf3d.update(tr['kf3d']['mean'], tr['kf3d']['covariance'], np.array(point3d))
            else:
                # Khởi tạo trạng thái 3D lần đầu tiên cho track này
                logger.info(f"Initiating 3D state for existing track {object_id}.")
                mean3d, cov3d = self.kf3d.initiate(np.array(point3d))
                tr['kf3d'] = {'mean': mean3d, 'covariance': cov3d}
        
        tr['last_frame_id'] = frame_id
        tr['time_since_update'] = 0

    def match(self, bbox, point3d, frame_id: int):
        """
        Kết hợp một phát hiện mới.
        - Nếu `point3d` tồn tại, sử dụng xếp tầng 2D->3D.
        - Nếu `point3d` là `None`, chỉ sử dụng kết hợp 2D (IoU cao nhất).
        """
        if not self.tracks:
            return None

        self.predict_all(frame_id)
        
        # --- Giai đoạn 1: Lọc 2D (Gating + Proximity) ---
        meas2d = self.bbox_tlbr_to_xywh(bbox)
        preliminary_candidates = []
        for pid, tr in self.tracks.items():
            # Gating bằng khoảng cách Mahalanobis 2D
            dist2d = self.kf2d.gating_distance(tr['kf2d']['mean'], tr['kf2d']['covariance'], meas2d.reshape(1, -1))[0]
            if dist2d > self.gating_thresh_2d:
                continue

            # Kiểm tra độ gần bằng IoU 2D
            pred_bbox = self.xywh_to_tlbr(tr['kf2d']['mean'][:4])
            iou_score = self.iou(pred_bbox, np.array(bbox, dtype=np.float32))
            if iou_score < self.proximity_thresh:
                continue
            
            preliminary_candidates.append((pid, iou_score))

        if not preliminary_candidates:
            return None

        # --- Giai đoạn 2: Lựa chọn dựa trên sự sẵn có của point3d ---
        
        # Trường hợp 1: Phát hiện có dữ liệu 3D -> Lựa chọn bằng 3D
        if point3d is not None:
            final_candidates = []
            for pid, iou_score in preliminary_candidates:
                tr = self.tracks[pid]
                # Chỉ xem xét các track đã có trạng thái 3D
                if 'kf3d' in tr:
                    dist3d = self.kf3d.gating_distance(tr['kf3d']['mean'], tr['kf3d']['covariance'], np.array(point3d))[0]
                    if dist3d <= self.gating_thresh_3d:
                        final_candidates.append((pid, dist3d))

            if not final_candidates:
                return None

            # Chọn ứng viên tốt nhất với khoảng cách 3D nhỏ nhất
            best_pid, _ = min(final_candidates, key=lambda x: x[1])
            return best_pid
        
        # Trường hợp 2: Phát hiện không có dữ liệu 3D -> Lựa chọn bằng 2D
        else:
            # Chọn ứng viên tốt nhất với điểm IoU cao nhất từ các ứng viên sơ bộ
            best_pid, _ = max(preliminary_candidates, key=lambda x: x[1])
            return best_pid


    def get_active_tracks(self):
        """Trả về danh sách ID của các track đang hoạt động."""
        return list(self.tracks.keys())


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the combined tracker
    tracker = TrackingManager(
        max_time_lost=10,
        proximity_thresh=0.4,
        gating_thresh_2d=chi2inv95[4],
        gating_thresh_3d=chi2inv95[3]
    )

    # Frame 1: A new person (ID 1) is detected
    bbox1 = [100, 100, 150, 200]  # [x1, y1, x2, y2]
    point1 = [5.0, 2.0, 0.0]      # [x, y, z] in meters
    tracker.add_track(object_id=1, bbox=bbox1, point3d=point1, frame_id=1)
    logging.info(f"Frame 1: Added track 1. Active tracks: {tracker.get_active_tracks()}")

    # Frame 2: A new detection appears. Is it person 1?
    new_bbox = [102, 103, 153, 204]
    new_point = [5.1, 2.05, 0.0]
    logging.info(f"\nFrame 2: Matching new detection (bbox={new_bbox}, point={new_point})")
    
    matched_id = tracker.match(bbox=new_bbox, point3d=new_point, frame_id=2)
    
    if matched_id is not None:
        logging.info(f"SUCCESS: Matched to track {matched_id}. Updating track.")
        tracker.update_track(object_id=matched_id, bbox=new_bbox, point3d=new_point, frame_id=2)
        # Print updated state
        updated_track = tracker.tracks[matched_id]
        logging.info(f"  -> Updated 2D mean (xywh): {np.round(updated_track['kf2d']['mean'][:4], 2)}")
        logging.info(f"  -> Updated 3D mean (xyz):  {np.round(updated_track['kf3d']['mean'][:3], 2)}")
    else:
        logging.warning("FAIL: No match found. This could become a new track.")