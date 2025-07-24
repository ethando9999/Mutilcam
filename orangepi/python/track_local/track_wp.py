import numpy as np
from collections import OrderedDict
import uuid
from scipy.optimize import linear_sum_assignment
import queue
from threading import Thread
import time 
import asyncio
from typing import Dict
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
        
        self.track_attributes: Dict[uuid.UUID, Dict] = {}


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
        stale_ids = [tid for tid, tr in self.tracks.items() if tr['time_since_update'] > self.max_time_lost]
        for tid in stale_ids:
            del self.tracks[tid]
            # <--- [SỬA LỖI] Xóa cả thuộc tính của track cũ để tránh memory leak --->
            if tid in self.track_attributes:
                del self.track_attributes[tid]
            logger.info(f"Removed stale track: {tid}")

    # <--- [TỐI ƯU] Các hàm _add và _update giờ nhận vào toàn bộ `person_data` --->
    def _add_track(self, person_data: Dict):
        new_id = uuid.uuid4()
        # Khởi tạo Kalman Filters
        mean_bbox, cov_bbox = self.kf_bbox.initiate(self.bbox_tlbr_to_xywh(person_data['bbox']))
        mean_wp, cov_wp = self.kf_wp.initiate(np.array(person_data['world_point_xy'], dtype=np.float32))
        
        # Lưu trạng thái động
        self.tracks[new_id] = {
            'mean_bbox': mean_bbox, 'covariance_bbox': cov_bbox,
            'mean_wp': mean_wp, 'covariance_wp': cov_wp,
            'time_since_update': 0, 'hits': 1
        }
        # Lưu trạng thái thuộc tính
        self.track_attributes[new_id] = person_data
        logger.info(f"Added new track: {new_id}")
        return new_id

    def _update_matched_track(self, tid: uuid.UUID, person_data: Dict):
        tr = self.tracks[tid]
        # Cập nhật Kalman Filters
        tr['mean_bbox'], tr['covariance_bbox'] = self.kf_bbox.update(tr['mean_bbox'], tr['covariance_bbox'], self.bbox_tlbr_to_xywh(person_data['bbox']))
        tr['mean_wp'], tr['covariance_wp'] = self.kf_wp.update(tr['mean_wp'], tr['covariance_wp'], np.array(person_data['world_point_xy'], dtype=np.float32))
        
        tr['time_since_update'] = 0
        tr['hits'] += 1
        # Cập nhật trạng thái thuộc tính với thông tin mới nhất
        self.track_attributes[tid] = person_data


    # <--- [TỐI ƯU] Hàm update giờ nhận vào list `people` trực tiếp --->
    def update(self, people_list: List[Dict]):
        self._predict_all_tracks()
        self._remove_stale_tracks()

        active_ids = list(self.tracks.keys())
        
        if not active_ids or not people_list:
            # Nếu không có track cũ, tạo mới cho tất cả detection
            return {f"det_{i}": self._add_track(p_data) for i, p_data in enumerate(people_list)}

        N, M = len(people_list), len(active_ids)
        cost = np.full((N, M), np.inf)

        # Tính toán ma trận chi phí
        for i, p_data in enumerate(people_list):
            bb = p_data['bbox']
            wp = np.array(p_data['world_point_xy']).reshape(1, 2)
            for j, tid in enumerate(active_ids):
                tr = self.tracks[tid]
                # Tính toán chi phí (logic giữ nguyên)
                d_wp = self.kf_wp.gating_distance(tr['mean_wp'], tr['covariance_wp'], wp)[0]
                if d_wp > self.gating_threshold_wp: continue
                d_bb = self.kf_bbox.gating_distance(tr['mean_bbox'], tr['covariance_bbox'], self.bbox_tlbr_to_xywh(bb).reshape(1, 4))[0]
                if d_bb > self.gating_threshold_bbox: continue
                pred_bb = self.xywh_to_tlbr(tr['mean_bbox'][:4])
                cost_iou = 1 - self.iou(pred_bb, bb)
                cost_maha = d_wp / self.gating_threshold_wp
                cost[i, j] = (self.iou_cost_weight * cost_iou + (1 - self.iou_cost_weight) * cost_maha)

        if not np.isfinite(cost).any():
             return {f"det_{i}": self._add_track(p_data) for i, p_data in enumerate(people_list)}

        rows, cols = linear_sum_assignment(cost)
        
        mapping = {}
        matched_dets = set()
        # Xử lý các cặp đã match
        for r, c in zip(rows, cols):
            if np.isfinite(cost[r, c]):
                tid = active_ids[c]
                self._update_matched_track(tid, people_list[r])
                mapping[f"det_{r}"] = tid
                matched_dets.add(r)
        
        # Xử lý các detection không match (tạo track mới)
        for r in range(N):
            if r not in matched_dets:
                new_tid = self._add_track(people_list[r])
                mapping[f"det_{r}"] = new_tid

        return mapping

    async def process_track(self, packet: dict):
        """
        Xử lý packet, chạy tracking và gửi đi payload cho từng người.
        Phiên bản này gọn hơn do hàm update đã được tối ưu.
        """
        camera_id = packet.get('camera_id')
        frame_id = packet.get('frame_id')
        time_detect = packet.get('time_detect')
        people = packet.get('people_list', [])
        
        # Chạy thuật toán tracking. Hàm update giờ nhận trực tiếp `people` list.
        mapping_from_tracker = self.update(people)
        
        # Nếu không có track nào, kết thúc sớm
        if not mapping_from_tracker:
            return

        # Lặp qua các track hợp lệ của frame hiện tại để tạo và gửi payload
        for track_id in mapping_from_tracker.values():
            try:
                # <--- [TỐI ƯU] Lấy thông tin thuộc tính trực tiếp từ tracker --->
                person_data = self.track_attributes[track_id]
                
                clothing = person_data.get('clothing_analysis') or {}
                gender = person_data.get('gender_analysis') or {}
                
                # Định dạng lại payload (logic giữ nguyên)
                regional_analysis = clothing.get("regional_analysis", {})
                torso_colors_raw = regional_analysis.get("torso_colors") or []
                pants_colors_raw = (regional_analysis.get("thigh_colors") or []) + \
                                   (regional_analysis.get("shin_colors") or [])

                torso_color_payload = [{"rgb": c[0][::-1], "percentage": round(c[1], 2)} for c in torso_colors_raw]
                pants_color_payload = [{"rgb": c[0][::-1], "percentage": round(c[1], 2)} for c in pants_colors_raw]

                final_person_payload = {
                    "frame_id": frame_id,
                    "person_id": str(track_id),
                    "gender": gender.get('gender', 'Unknown'),
                    "torso_color": torso_color_payload,
                    "torso_status": clothing.get("sleeve_type", "N/A"),
                    "pants_color": pants_color_payload,
                    "pants_status": clothing.get("pants_type", "N/A"),
                    "skin_tone": tuple(map(int, clothing["skin_tone"])) if clothing.get("skin_tone") is not None else None,
                    "height": round(person_data['est_height_m'], 2) if person_data.get('est_height_m') else None,
                    "time_detect": time_detect,
                    "camera_id": camera_id,
                    "world_point_xy": person_data.get('world_point_xy')
                }
                
                await self.processed_queue.put(final_person_payload)
                logger.info(f"[TRACK] Sent payload for person_id: {str(track_id)}")

            except KeyError:
                logger.warning(f"Track ID {track_id} found in mapping but not in attributes. Skipping.")
            except Exception as e:
                logger.error(f"Unexpected error while creating payload for track {track_id}: {e}", exc_info=True)

    async def run(self):
        logger.info(">>> Tracking task started. Waiting for detection batches...")
        while True:
            try:
                pkt = await self.detection_queue.get()
                if pkt is None:
                    break
                await self.process_track(pkt)
                self.detection_queue.task_done()
            except Exception as e:
                logger.error(f"Critical error in tracking loop: {e}", exc_info=True)
