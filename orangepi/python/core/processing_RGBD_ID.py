# file: python/core/processing_RGBD.py (Phiên bản cuối cùng - Đã sửa lỗi và tối ưu)

import asyncio
import cv2
import os
from datetime import datetime
import numpy as np

# Giả định các module này tồn tại và hoạt động đúng
from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
# from utils.pose_color_signature_new import PoseColorSignatureExtractor
# from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector_final import StereoProjector
from .keypoints_handle import get_head_center, get_torso_box, adjust_keypoints_to_box
from .height_estimator_pro import HeightEstimator
import config

logger = get_logger(__name__)

class FrameProcessor:
    """
    Xử lý các lô frame RGB-D, từ phát hiện người cho đến trích xuất thông tin chi tiết.
    """
    def __init__(self, calib_path: str, batch_size: int = 2):
        self.detector = HumanDetection()
        self.stereo_projector = StereoProjector(calib_file_path=calib_path)

        mtx_rgb = self.stereo_projector.params.get('mtx_rgb')
        if mtx_rgb is None: raise ValueError("mtx_rgb không có trong file hiệu chỉnh.")
        self.height_estimator = HeightEstimator(mtx_rgb)

        self.semaphore = asyncio.Semaphore(4)
        self.batch_size = batch_size

        self.debug_dir = os.path.join(config.OPI_CONFIG.get("results_dir", "results"), "debug_projection")
        os.makedirs(self.debug_dir, exist_ok=True)

        logger.info("FrameProcessor (Tối ưu Toàn diện) đã được khởi tạo.")

    # --------------------------------------------------------------------
    # CÁC HÀM PHỤ TRỢ (HELPERS)
    # --------------------------------------------------------------------

    # <<< SỬA LỖI: THÊM LẠI HÀM _is_detection_valid BỊ THIẾU >>>
    def _is_detection_valid(self, box: tuple, keypoints: np.ndarray, frame_shape: tuple) -> bool:
        """
        Xác thực chất lượng đầu vào từ YOLO.
        """
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        if np.any(np.isnan(box)) or np.any(np.isinf(box)): return False
        if x1 >= x2 or y1 >= y2: return False
        if x2 < 0 or y2 < 0 or x1 > w or y1 > h: return False

        if keypoints.shape[1] >= 3:
            valid_kpts_count = np.sum(keypoints[:, 2] > 0.3)
        else:
            valid_kpts_count = np.sum(np.sum(keypoints[:, :2], axis=1) > 0)

        if valid_kpts_count < 4:
            logger.debug(f"Phát hiện có ít keypoints ({valid_kpts_count}). Bỏ qua.")
            return False

        return True

    async def _run_in_executor(self, func, *args):
        return await asyncio.to_thread(func, *args)

    # --------------------------------------------------------------------
    # HÀM XỬ LÝ CHÍNH (CORE PROCESSING)
    # --------------------------------------------------------------------

    async def process_human_async(self, frame_id: int, box: tuple, keypoints: np.ndarray, tof_depth_map: np.ndarray):
        """
        Luồng xử lý tối ưu cho một người. Chỉ trả về dữ liệu cần thiết cho chiều cao.
        """
        async with self.semaphore:
            try:
                torso_box = get_torso_box(keypoints, box)
                distance_mm, status = await self._run_in_executor(self.stereo_projector.get_robust_distance, torso_box, tof_depth_map)

                if status != "OK" or distance_mm is None or not (100 < distance_mm < 4000):
                    return None

                distance_m = distance_mm / 1000.0
                est_height_m, height_status = await self._run_in_executor(self.height_estimator.estimate, keypoints, distance_m)

                if not est_height_m:
                    return None

                logger.info(f"✅✅ Xử lý người thành công: Khoảng cách={distance_m:.2f}m, Chiều cao={est_height_m:.2f}m ({height_status})")
                return {"est_height_m": est_height_m}
            except Exception as e:
                logger.error(f"Lỗi xử lý người cho frame {frame_id}: {e}", exc_info=True)
                return None

    async def process_frame_queue(self, frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
        """
        Vòng lặp chính: Lấy dữ liệu, điều phối tác vụ, và gửi dữ liệu theo từng frame.
        """
        frame_number = 0
        while True:
            batch_data_paths = []
            try:
                item = await asyncio.wait_for(frame_queue.get(), timeout=5.0)
                if item is None:
                    await processing_queue.put(None)
                    break
                batch_data_paths.append(item); frame_queue.task_done()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            while len(batch_data_paths) < self.batch_size:
                try:
                    item = frame_queue.get_nowait()
                    if item is None:
                        await processing_queue.put(None)
                        break
                    batch_data_paths.append(item); frame_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            if not batch_data_paths or any(item is None for item in batch_data_paths):
                 if any(item is None for item in batch_data_paths):
                    await processing_queue.put(None)
                 break

            try:
                loaded_data_map = {}
                frames_for_detection = []
                for i, (rgb_path, depth_path, amp_path) in enumerate(batch_data_paths):
                    try:
                        rgb_frame = cv2.imread(rgb_path)
                        depth_frame = np.load(depth_path) if depth_path and os.path.exists(depth_path) else None
                        if rgb_frame is not None and depth_frame is not None:
                            frames_for_detection.append(rgb_frame)
                            loaded_data_map[i] = (rgb_frame, depth_frame)
                    except Exception as e:
                        logger.error(f"Lỗi khi đọc file dữ liệu {rgb_path}: {e}")

                if not frames_for_detection: continue

                detection_results = await asyncio.gather(*[self._run_in_executor(self.detector.run_detection, f) for f in frames_for_detection], return_exceptions=True)

                all_human_tasks = []
                for i, detection_result in enumerate(detection_results):
                    if not detection_result or isinstance(detection_result, Exception): continue

                    original_rgb, original_depth = loaded_data_map[i]
                    keypoints_data, boxes_data = detection_result

                    # <<< TỐI ƯU: GỬI DỮ LIỆU ĐẾM NGƯỜI CHO TỪNG FRAME >>>
                    person_count_in_frame = len(boxes_data)
                    if person_count_in_frame > 0:
                        count_packet = {
                            "type": "person_count",
                            "data": {"total_person": person_count_in_frame}
                        }
                        await processing_queue.put(count_packet)
                        logger.info(f"Frame {frame_number + i}: Đã đưa {person_count_in_frame} người vào hàng đợi đếm.")

                    # Tạo các tác vụ xử lý chiều cao
                    for kpts, box in zip(keypoints_data, boxes_data):
                        if self._is_detection_valid(box, kpts, original_rgb.shape): # Dòng này giờ sẽ chạy được
                            all_human_tasks.append(
                                self.process_human_async(frame_number + i, box, kpts, original_depth)
                            )
                
                # Thu thập kết quả xử lý chiều cao và đóng gói
                if all_human_tasks:
                    processing_results = await asyncio.gather(*all_human_tasks, return_exceptions=True)
                    for result in processing_results:
                        if result and not isinstance(result, Exception):
                            height_packet = {
                                "type": "height_data",
                                "data": result
                            }
                            await processing_queue.put(height_packet)

                frame_number += len(frames_for_detection)
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong xử lý lô: {e}", exc_info=True)

async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, calib_path: str):
    logger.info("Khởi động worker xử lý (Phiên bản cuối cùng)...")
    try:
        processor = FrameProcessor(calib_path=calib_path, batch_size=2)
        await processor.process_frame_queue(frame_queue, processing_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True)