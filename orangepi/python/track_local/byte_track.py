import numpy as np
import uuid
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
import asyncio

from .kalman_filter_2D import KalmanFilter2D, chi2inv95
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class TrackingManager:
    """
    ByteTrack-inspired multi-object tracker:
      - High/low confidence split
      - Cascade IoU matching by track age
      - Mahalanobis gating
      - Score smoothing & decay
      - 8-D Kalman Filter motion model
    """
    def __init__(
        self,
        detection_queue: asyncio.Queue,
        processed_queue: asyncio.Queue,
        track_thresh: float = 0.5,
        max_time_lost: int = 30,
        iou_threshold: float = 0.6,
        ema_alpha: float = 0.9,
    ):
        self.detection_queue = detection_queue
        self.processed_queue = processed_queue
        self.kf = KalmanFilter2D()
        self.tracks = OrderedDict()

        self.track_thresh = track_thresh
        self.max_time_lost = max_time_lost
        self.iou_threshold = iou_threshold
        self.ema_alpha = ema_alpha

    @staticmethod
    def bbox_tlbr_to_xywh(bbox):
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w/2., y1 + h/2., w, h], dtype=np.float32)

    @staticmethod
    def xywh_to_tlbr(xywh):
        cx, cy, w, h = xywh[:4]
        return np.array([cx - w/2., cy - h/2., cx + w/2., cy + h/2.], dtype=np.float32)

    @staticmethod
    def iou(b1, b2):
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        iw = max(0., xi2 - xi1); ih = max(0., yi2 - yi1)
        inter = iw * ih
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter/union if union>0 else 0.

    def _predict_all(self):
        if not self.tracks:
            return
        means = np.stack([tr['mean'] for tr in self.tracks.values()])
        covs = np.stack([tr['cov'] for tr in self.tracks.values()])
        pred_means, pred_covs = self.kf.multi_predict(means, covs)
        for tid, (m, c) in zip(self.tracks.keys(), zip(pred_means, pred_covs)):
            tr = self.tracks[tid]
            tr['mean'], tr['cov'] = m, c
            tr['age'] += 1
            tr['time_since_update'] += 1

    def _gated_cost(self, det, tr):
        meas = self.bbox_tlbr_to_xywh(det['bbox'])
        # Mahalanobis gating
        gating_dist = self.kf.gating_distance(
            tr['mean'], tr['cov'], meas[None, :], only_position=False)
        if gating_dist[0] > chi2inv95[4]:
            return np.inf
        # IoU gating & cost
        pred_bbox = self.xywh_to_tlbr(tr['mean'][:4])
        iou_val = self.iou(det['bbox'], pred_bbox)
        if iou_val < self.iou_threshold:
            return np.inf
        return 1 - iou_val

    def _associate(self, dets, track_ids):
        if not dets or not track_ids:
            return [], list(dets.keys()), list(track_ids)

        det_keys = list(dets.keys())
        T = len(track_ids)

        # 1) Build full cost matrix
        D = len(det_keys)
        cost = np.full((D, T), np.inf, dtype=np.float32)
        for i, d in enumerate(det_keys):
            for j, t in enumerate(track_ids):
                cost[i, j] = self._gated_cost(dets[d], self.tracks[t])

        # 2) Find which rows/columns have any finite entries
        valid_row = np.any(np.isfinite(cost), axis=1)
        valid_col = np.any(np.isfinite(cost), axis=0)

        # Map to original indices
        row_idx = np.nonzero(valid_row)[0]
        col_idx = np.nonzero(valid_col)[0]

        # Rows/cols with *no* feasible match become unmatched immediately
        unmatched_rows = set(det_keys[i] for i in range(D) if not valid_row[i])
        unmatched_cols = set(track_ids[j] for j in range(T) if not valid_col[j])

        # If no feasible pairs at all, bail out
        if len(row_idx) == 0 or len(col_idx) == 0:
            return [], list(dets.keys()), list(track_ids)

        # 3) Solve on the sub‐matrix
        sub_cost = cost[np.ix_(row_idx, col_idx)]
        rows_sub, cols_sub = linear_sum_assignment(sub_cost)

        # 4) Map assignments back
        matches = []
        for r_sub, c_sub in zip(rows_sub, cols_sub):
            # Only accept finite costs (gating may have put inf in the submatrix)
            if np.isfinite(sub_cost[r_sub, c_sub]):
                det_id = det_keys[row_idx[r_sub]]
                tr_id  = track_ids[col_idx[c_sub]]
                matches.append((det_id, tr_id))

        # Collect the leftovers
        matched_d = {d for d, _ in matches}
        matched_t = {t for _, t in matches}
        unmatched_rows |= set(det_keys) - matched_d
        unmatched_cols |= set(track_ids) - matched_t

        return matches, list(unmatched_rows), list(unmatched_cols)

    def update(self, detections: dict):
        # 1) Predict
        self._predict_all()

        # 2) Split high/low
        det_high = {k: v for k, v in detections.items() if v['bbox'][4] >= self.track_thresh}
        det_low  = {k: v for k, v in detections.items() if v['bbox'][4] <  self.track_thresh}

        # 3) Get active & lost track IDs
        active = [tid for tid, tr in self.tracks.items() if tr['time_since_update'] <= 1]
        lost   = [tid for tid, tr in self.tracks.items() if 1 < tr['time_since_update'] <= self.max_time_lost]

        results = {}
        # 4) Cascade match high-confidence by track age
        for age in range(1, min(4, self.max_time_lost) + 1):
            cand = [tid for tid in active if self.tracks[tid]['age'] == age]
            if not cand or not det_high:
                continue
            matches, un_d, un_t = self._associate(
                {k: det_high[k] for k in det_high}, cand)
            for d, t in matches:
                det = det_high[d]
                tr = self.tracks[t]
                meas = self.bbox_tlbr_to_xywh(det['bbox'])
                tr['mean'], tr['cov'] = self.kf.update(tr['mean'], tr['cov'], meas)
                tr['time_since_update'] = 0
                tr['hits'] += 1
                # EMA score update
                tr['score'] = self.ema_alpha * det['bbox'][4] + (1 - self.ema_alpha) * tr['score']
                tr['world_point'] = det['world_point']
                results[d] = t
                det_high.pop(d, None)
            # update active pool
            active = [tid for tid in active if tid not in un_t]

        # 5) Stage 2: rescue with low-confidence
        matches_low, un_low, un_act2 = self._associate(det_low, active)
        for d, t in matches_low:
            det = det_low[d]
            tr = self.tracks[t]
            meas = self.bbox_tlbr_to_xywh(det['bbox'])
            tr['mean'], tr['cov'] = self.kf.update(tr['mean'], tr['cov'], meas)
            tr['time_since_update'] = 0
            tr['hits'] += 1
            tr['score'] = self.ema_alpha * det['bbox'][4] + (1 - self.ema_alpha) * tr['score']
            tr['world_point'] = det['world_point']
            results[d] = t
            det_low.pop(d, None)

        # 6) Revival on lost tracks
        matches_rev, det_high2, _ = self._associate(det_high, lost)
        for d, t in matches_rev:
            det = det_high[d]
            tr = self.tracks[t]
            meas = self.bbox_tlbr_to_xywh(det['bbox'])
            tr['mean'], tr['cov'] = self.kf.update(tr['mean'], tr['cov'], meas)
            tr['time_since_update'] = 0
            tr['hits'] += 1
            tr['score'] = self.ema_alpha * det['bbox'][4] + (1 - self.ema_alpha) * tr['score']
            tr['world_point'] = det['world_point']
            results[d] = t
            det_high.pop(d, None)

        # 7) Create new tracks for remaining high-confidence
        for d, det in list(det_high.items()):
            meas = self.bbox_tlbr_to_xywh(det['bbox'])
            mean, cov = self.kf.initiate(meas)
            tid = uuid.uuid4()
            self.tracks[tid] = {
                'mean': mean,
                'cov': cov,
                'age': 0,
                'time_since_update': 0,
                'hits': 1,
                'score': det['bbox'][4],
                'world_point': det['world_point'],
            }
            results[d] = tid

        # 8) Remove stale tracks
        to_del = [tid for tid, tr in self.tracks.items()
                  if tr['time_since_update'] > self.max_time_lost]
        for tid in to_del:
            del self.tracks[tid]
            logger.info(f"Removed stale track: {tid}")

        return results

    async def process_track(self, packet: dict):
        camera_id = packet.get('camera_id')
        frame_id = packet.get('frame_id')
        time_detect = packet.get('time_detect')
        people = packet.get('people_list', [])

        dets = { 
            f"det_{i}": {
                'bbox': p['bbox'],
                'world_point': p['world_point_xy']
            }
            for i, p in enumerate(people)
        }

        mapping = self.update(dets)
        mapping_list = [
            {"id": str(tid), "world_point": dets[d]['world_point']}
            for d, tid in mapping.items()
        ]

        result_packet = {
            'camera_id': camera_id,
            'frame_id': frame_id,
            'time_detect': time_detect,
            'total_detect': len(people),
            'mapping': mapping_list,
        }
        await self.processed_queue.put(result_packet)
        logger.info(
            f"[TRACK] Đã put vào track_queue: {result_packet}"
        )

    async def run(self):
        logger.info("Tracking task started...")
        while True:
            pkt = await self.detection_queue.get()
            if pkt is None:
                logger.info("Shutting down TrackingManager.")
                break
            try:
                await self.process_track(pkt)
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}", exc_info=True)
            finally:
                self.detection_queue.task_done()
