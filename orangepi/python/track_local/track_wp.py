import numpy as np
from collections import OrderedDict
import uuid
from scipy.optimize import linear_sum_assignment
import queue
from threading import Thread
import time 
import asyncio

# Giả sử các lớp này đã được định nghĩa ở nơi khác trong dự án của bạn
from .kalman_filter_2D import KalmanFilter2D, chi2inv95 as chi2inv95_bbox
from .kalman_filter_world_points import KalmanFilterWorldPoint, chi2inv95 as chi2inv95_wp
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Ngưỡng quantile 0.95 của phân phối Chi-square
CHI2INV95_4_BBOX = chi2inv95_bbox[4]
CHI2INV95_2_WP = chi2inv95_wp[2]

class TrackingManager:
    """
    Quản lý theo dõi đa đối tượng bằng cách kết hợp Kalman Filter trên cả không gian ảnh (bbox)
    và không gian thế giới (world point), tương tự triết lý của BoT-SORT.
    Sử dụng thuật toán Hungarian để giải quyết bài toán gán ghép tối ưu.
    """
    def __init__(
        self,
        detection_queue: asyncio.Queue,
        processed_queue: asyncio.Queue,
        gating_threshold_bbox: float = CHI2INV95_4_BBOX,
        gating_threshold_wp: float = CHI2INV95_2_WP,
        max_time_lost: int = 30,
        iou_cost_weight: float = 0.5
    ):
        self.detection_queue = detection_queue
        self.processed_queue = processed_queue

        self.kf_bbox = KalmanFilter2D()
        self.kf_wp = KalmanFilterWorldPoint()
        self.tracks = OrderedDict()

        self.gating_threshold_bbox = gating_threshold_bbox
        self.gating_threshold_wp = gating_threshold_wp
        self.max_time_lost = max_time_lost
        self.iou_cost_weight = iou_cost_weight

    @staticmethod
    def bbox_tlbr_to_xywh(bbox):
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w/2., y1 + h/2., w, h], dtype=np.float32)

    @staticmethod
    def xywh_to_tlbr(xywh):
        cx, cy, w, h = xywh
        return np.array([cx - w/2., cy - h/2., cx + w/2., cy + h/2.], dtype=np.float32)

    @staticmethod
    def iou(b1, b2):
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        iw = max(0., xi2 - xi1); ih = max(0., yi2 - yi1)
        inter = iw * ih
        union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
        return inter / union if union > 0 else 0.0

    def _predict_all_tracks(self):
        for tr in self.tracks.values():
            tr['mean_bbox'], tr['covariance_bbox'] = self.kf_bbox.predict(
                tr['mean_bbox'], tr['covariance_bbox'])
            tr['mean_wp'], tr['covariance_wp'] = self.kf_wp.predict(
                tr['mean_wp'], tr['covariance_wp'])
            tr['time_since_update'] += 1

    def _remove_stale_tracks(self):
        stale_ids = [tid for tid, tr in self.tracks.items()
                     if tr['time_since_update'] > self.max_time_lost]
        for tid in stale_ids:
            del self.tracks[tid]
            logger.info(f"Removed stale track: {tid}")

    def _add_track(self, bbox, world_point):
        new_id = uuid.uuid4()
        meas_bbox = self.bbox_tlbr_to_xywh(bbox)
        mean_bbox, cov_bbox = self.kf_bbox.initiate(meas_bbox)
        mean_wp, cov_wp = self.kf_wp.initiate(
            np.array(world_point, dtype=np.float32))
        self.tracks[new_id] = {
            'mean_bbox': mean_bbox, 'covariance_bbox': cov_bbox,
            'mean_wp': mean_wp, 'covariance_wp': cov_wp,
            'time_since_update': 0, 'hits': 1
        }
        logger.info(f"Added new track: {new_id}")
        return new_id

    def _update_matched_track(self, tid, bbox, world_point):
        tr = self.tracks[tid]
        meas_bbox = self.bbox_tlbr_to_xywh(bbox)
        tr['mean_bbox'], tr['covariance_bbox'] = self.kf_bbox.update(
            tr['mean_bbox'], tr['covariance_bbox'], meas_bbox)
        meas_wp = np.array(world_point, dtype=np.float32)
        tr['mean_wp'], tr['covariance_wp'] = self.kf_wp.update(
            tr['mean_wp'], tr['covariance_wp'], meas_wp)
        tr['time_since_update'] = 0
        tr['hits'] += 1

    def update(self, detections: dict):
        # 1) Dự đoán và dọn dẹp
        self._predict_all_tracks()
        self._remove_stale_tracks()

        active_ids = list(self.tracks.keys())
        det_ids = list(detections.keys())
        # Nếu không có track cũ hoặc không có detections
        if not active_ids or not det_ids:
            results = {}
            for d in det_ids:
                results[d] = self._add_track(
                    detections[d]['bbox'], detections[d]['world_point'])
            return results

        # 2) Tạo ma trận chi phí
        N, M = len(det_ids), len(active_ids)
        cost = np.full((N, M), np.inf)
        for i, d in enumerate(det_ids):
            bb = detections[d]['bbox']
            wp = np.array(detections[d]['world_point']).reshape(1, 2)
            for j, tid in enumerate(active_ids):
                tr = self.tracks[tid]
                d_wp = self.kf_wp.gating_distance(
                    tr['mean_wp'], tr['covariance_wp'], wp)[0]
                if d_wp > self.gating_threshold_wp:
                    continue
                d_bb = self.kf_bbox.gating_distance(
                    tr['mean_bbox'], tr['covariance_bbox'],
                    self.bbox_tlbr_to_xywh(bb).reshape(1, 4))[0]
                if d_bb > self.gating_threshold_bbox:
                    continue
                pred_bb = self.xywh_to_tlbr(tr['mean_bbox'][:4])
                cost_iou = 1 - self.iou(pred_bb, bb)
                cost_maha = d_wp / self.gating_threshold_wp
                cost[i, j] = (self.iou_cost_weight * cost_iou +
                              (1 - self.iou_cost_weight) * cost_maha)

        # Nếu không có cặp nào khả thi, tạo track mới cho tất cả
        if not np.isfinite(cost).any():
            results = {}
            for d in det_ids:
                results[d] = self._add_track(
                    detections[d]['bbox'], detections[d]['world_point'])
            return results

        # 3) Gán ghép Hungarian
        rows, cols = linear_sum_assignment(cost)
        results = {}
        matched_r, matched_c = set(), set()
        for r, c in zip(rows, cols):
            if np.isfinite(cost[r, c]):
                det, tid = det_ids[r], active_ids[c]
                self._update_matched_track(
                    tid, detections[det]['bbox'], detections[det]['world_point'])
                results[det] = tid
                matched_r.add(r)
                matched_c.add(c)

        # 4) Track mới cho unmatched detections
        for r in set(range(N)) - matched_r:
            det = det_ids[r]
            results[det] = self._add_track(
                detections[det]['bbox'], detections[det]['world_point'])

        return results

    def get_active_tracks(self):
        """Chuyển đổi sang định dạng JSON-serializable."""
        data = {}
        for tid, tr in self.tracks.items():
            bbox = self.xywh_to_tlbr(tr['mean_bbox'][:4]).tolist()
            wp = tr['mean_wp'][:2].tolist()
            data[str(tid)] = {
                'bbox_tlbr': bbox,
                'world_point': wp,
                'time_since_update': tr['time_since_update'],
                'hits': tr['hits']
            }
        return data

    async def process_track(self, packet: dict):
        """
        Xử lý một gói dữ liệu từ detection_queue, thực hiện tracking,
        và đưa kết quả vào processed_queue.
        Packet kết quả bao gồm camera_id, frame_id, time_detect, total_detect,
        và một list 'mapping' các đối tượng đã được gán ID.
        """
        # Bước 1: Trích xuất thông tin cơ bản từ packet đầu vào.
        # Giả định packet đầu vào có chứa 'camera_id'.
        camera_id = packet.get('camera_id')
        frame_id = packet.get('frame_id')
        time_detect = packet.get('time_detect')
        people = packet.get('people_list', [])
        
        # Bước 2: Tính toán tổng số detection trong frame này.
        total_detect = len(people)
        
        # Bước 3: Chuẩn bị dữ liệu detection cho hàm update của tracker.
        dets = {
            f"det_{i}": {
                'bbox': p['bbox'], 
                'world_point': p['world_point_xy'] 
            } 
            for i, p in enumerate(people)
        }
        
        # Bước 4: Chạy thuật toán tracking để nhận về dictionary ánh xạ.
        mapping_from_tracker = self.update(dets)
        
        # Bước 5: Chuyển đổi từ dictionary sang định dạng list mong muốn.
        mapping_list = [
            {
                "id": str(track_id),
                "world_point": dets[det_id]['world_point']
            }
            for det_id, track_id in mapping_from_tracker.items()
        ]
        
        # Bước 6: Tạo gói kết quả cuối cùng với tất cả các trường yêu cầu.
        result_packet = {
            'camera_id': camera_id,         # <-- Trường mới
            'frame_id': frame_id,
            'time_detect': time_detect,
            'total_detect': total_detect,   # <-- Trường mới
            'mapping': mapping_list
        }
        
        # Bước 7: Đưa gói dữ liệu vào queue và ghi log.
        await self.processed_queue.put(result_packet)
        logger.info(f"[TRACK] đã put vào processed_queue: {result_packet}")

    async def run(self):
        print(">>> Tracking task started. Waiting for detection batches...")
        while True:
            pkt = await self.detection_queue.get()
            if pkt is None:
                break
            await self.process_track(pkt)
            self.detection_queue.task_done()