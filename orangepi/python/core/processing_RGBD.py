# file: python/core/processing_RGBD.py (Phiên bản cuối cùng - Tích hợp logic gửi định kỳ)

import asyncio
import cv2
import os
import time
import numpy as np

from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector_final import StereoProjector
from .keypoints_handle import get_torso_box, adjust_keypoints_to_box
from .height_estimator import HeightEstimator 
import config

logger = get_logger(__name__)

class FrameProcessor:
    def __init__(self, processing_queue: asyncio.Queue, socket_queue: asyncio.Queue, batch_size: int = 2):
        self.detector = HumanDetection()
        self.stereo_projector = StereoProjector()
        
        mtx_rgb = self.stereo_projector.params.get('mtx_rgb')
        if mtx_rgb is None: raise ValueError("mtx_rgb không có trong file hiệu chỉnh.")
        self.height_estimator = HeightEstimator(mtx_rgb)
        
        self.processing_queue = processing_queue
        self.socket_queue = socket_queue
        self.table_id = config.OPI_CONFIG.get("SOCKET_TABLE_ID", 1)
        
        self.semaphore = asyncio.Semaphore(4)
        self.batch_size = batch_size
        
        # LOGIC MỚI: Thêm các biến trạng thái để gửi định kỳ
        self.last_person_count = 0
        self.last_zero_sent_time = time.time()
        self.periodic_send_interval = 10 # Giây
        
        logger.info("FrameProcessor (Logic gửi định kỳ) đã được khởi tạo.")

    def _is_detection_valid(self, box: tuple, keypoints: np.ndarray, frame_shape: tuple) -> bool:
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        if np.any(np.isnan(box)) or np.any(np.isinf(box)) or x1 >= x2 or y1 >= y2 or x2 < 0 or y2 < 0 or x1 > w or y1 > h:
            return False
        
        if keypoints.shape[1] >= 3:
            valid_kpts_count = np.sum(keypoints[:, 2] > 0.3)
        else:
            valid_kpts_count = np.sum(np.any(keypoints > 0, axis=1))
        
        if valid_kpts_count < 4:
            return False
        return True

    async def _run_in_executor(self, func, *args):
        return await asyncio.to_thread(func, *args)

    async def process_human_async(self, rgb_frame: np.ndarray, tof_depth_map: np.ndarray, box: tuple, keypoints: np.ndarray):
        async with self.semaphore:
            try:
                torso_box = get_torso_box(keypoints, box)
                distance_mm, status = await self._run_in_executor(self.stereo_projector.get_robust_distance, torso_box, tof_depth_map)

                if status != "OK" or distance_mm is None or not (800 < distance_mm < 3200):
                    return None

                distance_m = distance_mm / 1000.0
                est_height_m, height_status = await self._run_in_executor(self.height_estimator.estimate, keypoints, distance_m)
                
                human_box_img = rgb_frame[box[1]:box[3], box[0]:box[2]]
                if human_box_img.size == 0: return None
                
                adjusted_keypoints = adjust_keypoints_to_box(keypoints, box)
                body_parts = await self._run_in_executor(extract_body_parts_from_frame, human_box_img, adjusted_keypoints)

                return {
                    "human_box": human_box_img, "body_parts": body_parts, "bbox": box,
                    "map_keypoints": keypoints, "est_height_m": est_height_m, "height_status": height_status,
                    "distance_mm": distance_mm, "body_color": None
                }
            except Exception as e:
                logger.error(f"Lỗi xử lý 1 người: {e}", exc_info=True)
                return None

    async def _send_zero_count_periodically(self, force_send=False):
        """Hàm helper để gửi số người là 0."""
        now = time.time()
        time_since_last_send = now - self.last_zero_sent_time
        
        if force_send or time_since_last_send > self.periodic_send_interval:
            if force_send:
                logger.info("Chuyển trạng thái sang không có người. Gửi số người = 0.")
            else:
                logger.info(f"Đã {self.periodic_send_interval}s không có người, gửi định kỳ số người = 0.")
            
            await self.socket_queue.put({"type": "person_count", "data": {"total_person": 0}})
            self.last_zero_sent_time = now

    async def process_frame_queue(self, frame_queue: asyncio.Queue):
        frame_number = 0
        while True:
            try:
                # LOGIC MỚI: Xử lý timeout để gửi định kỳ
                batch_data_paths = [await asyncio.wait_for(frame_queue.get(), timeout=1.0)] # Giảm timeout để check thường xuyên hơn
                if batch_data_paths[0] is None: break
                frame_queue.task_done()
            except asyncio.TimeoutError:
                # Khi không có frame mới, kiểm tra xem có cần gửi định kỳ số 0 không
                if self.last_person_count == 0:
                    await self._send_zero_count_periodically()
                else: # Chuyển từ có người -> không có người do hết frame
                    await self._send_zero_count_periodically(force_send=True)
                    self.last_person_count = 0
                continue

            # Nạp thêm frame vào lô nếu có
            while len(batch_data_paths) < self.batch_size:
                try:
                    item = frame_queue.get_nowait()
                    if item is None: frame_queue.put_nowait(None); break
                    batch_data_paths.append(item); frame_queue.task_done()
                except asyncio.QueueEmpty: break

            try:
                loaded_data_map, frames_for_detection = {}, []
                for i, (rgb_path, depth_path, _) in enumerate(batch_data_paths):
                    try:
                        rgb, depth = cv2.imread(rgb_path), np.load(depth_path)
                        if rgb is not None and depth is not None:
                            frames_for_detection.append(rgb); loaded_data_map[i] = (rgb, depth)
                    except Exception as e: logger.error(f"Lỗi đọc file {rgb_path}: {e}")
                
                if not frames_for_detection: continue

                detection_results = await asyncio.gather(*[self._run_in_executor(self.detector.run_detection, f) for f in frames_for_detection])
                
                human_tasks = []
                for i, res in enumerate(detection_results):
                    if not res or res[0] is None or len(res[0]) == 0: continue
                    original_rgb, original_depth = loaded_data_map[i]
                    keypoints_data, boxes_data = res
                    for kpts, box in zip(keypoints_data, boxes_data):
                        if self._is_detection_valid(box, kpts, original_rgb.shape):
                            human_tasks.append(self.process_human_async(original_rgb, original_depth, box, kpts))

                person_results = await asyncio.gather(*human_tasks) if human_tasks else []
                
                all_valid_heights_cm = []
                valid_persons_for_reid = [p for p in person_results if p]

                current_total_persons = len(valid_persons_for_reid)

                # --- LOGIC MỚI: Xử lý gửi dựa trên trạng thái ---
                if current_total_persons > 0:
                    # Khi có người, gửi ngay lập tức
                    logger.info(f"Phát hiện {current_total_persons} người.")
                    await self.socket_queue.put({
                        "type": "person_count", "data": {"total_person": current_total_persons}
                    })
                    
                    # Lấy danh sách chiều cao và gửi đi
                    for person_data in valid_persons_for_reid:
                        if person_data.get("est_height_m") is not None:
                            height_cm = person_data["est_height_m"] * 100.0
                            if 150.0 <= height_cm <= 190.0:
                                all_valid_heights_cm.append(round(height_cm, 2))
                    
                    if all_valid_heights_cm:
                        await self.socket_queue.put({
                            "type": "height_data",
                            "data": {"table_id": self.table_id, "heights_cm": all_valid_heights_cm}
                        })
                else: # current_total_persons == 0
                    # Nếu lần trước có người, giờ về 0 -> gửi ngay
                    if self.last_person_count > 0:
                        await self._send_zero_count_periodically(force_send=True)
                    else: # Nếu lần trước cũng là 0 -> kiểm tra chu kỳ 10s
                        await self._send_zero_count_periodically()

                self.last_person_count = current_total_persons
                
                # GỬI DỮ LIỆU TỚI LUỒNG RE-ID (CHI TIẾT)
                for i, reid_data in enumerate(valid_persons_for_reid):
                    reid_data["camera_id"] = config.OPI_CONFIG.get("device_id", "opi_01")
                    reid_data["frame_id"] = frame_number + i
                    await self.processing_queue.put(reid_data)
                
                frame_number += len(frames_for_detection)
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong xử lý lô: {e}", exc_info=True)

async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, socket_queue: asyncio.Queue):
    logger.info("Khởi động worker xử lý (phiên bản gửi định kỳ)...")
    try:
        processor = FrameProcessor(
            processing_queue=processing_queue,
            socket_queue=socket_queue,
            batch_size=2
        ) 
        await processor.process_frame_queue(frame_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True)