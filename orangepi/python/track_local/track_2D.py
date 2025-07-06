import numpy as np
from collections import OrderedDict, deque

# Assuming these are defined elsewhere in your project
from .kalman_filter_2D import KalmanFilter2D, chi2inv95
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Chi-square 0.95 quantile with 4 degrees of freedom (for bbox [cx, cy, w, h])
CHI2INV95_4 = chi2inv95[4]


class TrackingManager:
    """
    Manages track state for each person_id using a 2D Kalman Filter (8-d state).
    This version relies solely on motion (position and velocity) for tracking.

    Each track stores:
      - mean (8-d state vector), covariance (8x8 matrix)
      - last_frame_id, time_since_update
    """
    def __init__(self,
                 gating_threshold: float = CHI2INV95_4,
                 max_time_lost: int = 30,
                 proximity_thresh: float = 0.7):
        """
        Args:
            gating_threshold (float): The squared Mahalanobis distance threshold for gating.
            max_time_lost (int): Maximum number of frames a track can be lost before being deleted.
            proximity_thresh (float): Minimum IoU threshold for a detection to be matched with a track.
        """
        self.kf = KalmanFilter2D()
        # tracks: OrderedDict to store active tracks.
        # Key: person_id, Value: dict containing track state.
        self.tracks = OrderedDict()
        self.gating_threshold = gating_threshold
        self.max_time_lost = max_time_lost
        self.proximity_thresh = proximity_thresh

    @staticmethod
    def bbox_tlbr_to_xywh(bbox):
        """
        Converts a bounding box from [x1, y1, x2, y2] to [cx, cy, w, h].
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
        Converts a bounding box from [cx, cy, w, h] to [x1, y1, x2, y2].
        """
        cx, cy, w, h = xywh
        x1 = cx - w / 2.
        y1 = cy - h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def iou(b1, b2):
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
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

    def predict_all(self, current_frame_id: int):
        """
        Predicts the state of all tracks to the current frame.
        It increments the time_since_update counter for each track and removes stale tracks.
        """
        for pid, tr in self.tracks.items():
            last = tr['last_frame_id']
            if current_frame_id > last:
                # Predict one step forward
                mean, cov = tr['mean'], tr['covariance']
                new_mean, new_cov = self.kf.predict(mean.copy(), cov.copy())
                tr['mean'], tr['covariance'] = new_mean, new_cov
                tr['last_frame_id'] = current_frame_id
                tr['time_since_update'] += 1
        self._remove_stale_tracks()

    def _remove_stale_tracks(self):
        """
        Removes tracks that have exceeded the maximum lost time.
        """
        to_remove = [pid for pid, tr in self.tracks.items()
                     if tr['time_since_update'] > self.max_time_lost]
        for pid in to_remove:
            del self.tracks[pid]

    def add_track(self, person_id, bbox, frame_id: int):
        """
        Initializes and adds a new track for a given person_id and bounding box.
        If the track already exists, it will be reset.
        """
        measurement = self.bbox_tlbr_to_xywh(bbox)
        mean, cov = self.kf.initiate(measurement)
        self.tracks[person_id] = {
            'mean': mean,
            'covariance': cov,
            'last_frame_id': frame_id,
            'time_since_update': 0,
        }

    def update_track(self, person_id, bbox, frame_id: int):
        """
        Updates an existing track with a new measurement (bbox).
        If the track does not exist, it calls add_track to create it.
        """
        if person_id not in self.tracks:
            self.add_track(person_id, bbox, frame_id)
            return
            
        tr = self.tracks[person_id]
        
        # Predict if the track was not updated in this frame
        if frame_id > tr['last_frame_id']:
            mean, cov = tr['mean'], tr['covariance']
            new_mean, new_cov = self.kf.predict(mean.copy(), cov.copy())
            tr['mean'], tr['covariance'] = new_mean, new_cov
            tr['last_frame_id'] = frame_id
            tr['time_since_update'] += 1
            
        # Update with the new measurement
        measurement = self.bbox_tlbr_to_xywh(bbox)
        mean, cov = tr['mean'], tr['covariance']
        new_mean, new_cov = self.kf.update(mean, cov, measurement)
        tr['mean'], tr['covariance'] = new_mean, new_cov
        tr['last_frame_id'] = frame_id
        tr['time_since_update'] = 0

    def match(self, bbox, frame_id: int):
        """
        Matches a detection (bbox) with existing tracks.

        The process is as follows:
        1. Predict all tracks to the current frame_id.
        2. Perform spatial gating using Mahalanobis distance to find candidate tracks.
        3. For candidates, calculate IoU with the detection bbox; keep those above proximity_thresh.
        4. Select the best match based on the highest IoU score.

        Returns:
            The person_id of the best matching track, or None if no suitable match is found.
        """
        if not self.tracks:
            logger.warning("No existing tracks to match against.")
            return None

        # 1. Predict all tracks to the current frame
        self.predict_all(frame_id)

        # 2. Gating based on Mahalanobis distance
        meas = self.bbox_tlbr_to_xywh(bbox).reshape(1, 4)
        candidates = []
        for pid, tr in self.tracks.items():
            dist = self.kf.gating_distance(tr['mean'], tr['covariance'], meas,
                                            only_position=False, metric='maha')[0]
            if dist <= self.gating_threshold:
                candidates.append(pid)
        
        if not candidates:
            return None

        # 3. Filter candidates by IoU proximity
        iou_candidates = []
        for pid in candidates:
            predicted_bbox_tlbr = self.xywh_to_tlbr(self.tracks[pid]['mean'][:4])
            iou_score = self.iou(predicted_bbox_tlbr, np.array(bbox, dtype=np.float32))
            if iou_score >= self.proximity_thresh:
                iou_candidates.append((pid, iou_score))
        
        if not iou_candidates:
            return None

        # 4. Select the best candidate based on highest IoU
        best_candidate = max(iou_candidates, key=lambda x: x[1])
        best_pid = best_candidate[0]

        return best_pid

    def get_active_tracks(self):
        """
        Returns a list of person_ids for all currently active (non-stale) tracks.
        """
        return list(self.tracks.keys())