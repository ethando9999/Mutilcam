# file: python/core/processing_RGBD_final.py

import asyncio
import cv2
import time
import numpy as np
from typing import Dict, Any

# --- Import các thành phần cần thiết ---
from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector_final import StereoProjectorFinal
from .height_estimator_pro import HeightEstimatorPro
from utils.latest_queue import LatestFrameQueue # Sửa lỗi chính tả "lastest"

# <<< SỬA LỖI IMPORT: Trỏ đến đúng file chứa lớp Track3DPro >>>
from tracking.track_3d_pro import Track3DPro

logger = get_logger(__name__)

# --- Cấu hình cho Worker ---
NUM_SLOW_WORKERS = 2  # Số worker xử lý sâu chạy song song. Tinh chỉnh dựa trên CPU/NPU.
FAST_LOOP_LOG_INTERVAL = 10.0 # Log FPS của vòng lặp nhanh mỗi 10 giây.

class FrameProcessorFinal:
    """
    Triển khai kiến trúc xử lý hai tầng với hiệu năng và độ ổn định cao.
    - Vòng lặp nhanh: Phát hiện và theo dõi vị trí.
    - Worker chậm: Xử lý sâu các đặc trưng.
    """
    def __init__(self, config: Dict, tracker: Track3DPro):
        self.config = config
        self.tracker = tracker # Sử dụng tracker đã được khởi tạo từ run_pipeline.py

        # Khởi tạo các thành phần xử lý
        self.detector = HumanDetection()
        calib_path = config.get("calib_file_path")
        self.stereo_projector = StereoProjectorFinal(calib_file_path=calib_path)
        self.height_estimator = HeightEstimatorPro(stereo_projector=self.stereo_projector)
        self.pose_processor = PoseColorSignatureExtractor()

        # Hàng đợi nội bộ kết nối giữa Fast Loop và Slow Workers
        self.reid_task_queue = asyncio.Queue(maxsize=100)
        # Semaphore để giới hạn số tác vụ nặng chạy đồng thời, tránh quá tải
        self.slow_task_semaphore = asyncio.Semaphore(NUM_SLOW_WORKERS)
        
        logger.info(f"FrameProcessorFinal đã khởi tạo với {NUM_SLOW_WORKERS} worker chậm.")

    async def run_fast_loop(self, frame_queue: LatestFrameQueue):
        """Vòng lặp nhanh: Nhận frame, chạy YOLO, cập nhật tracker, và gửi tác vụ."""
        logger.info("🚀 Vòng lặp nhanh (Fast Loop) đã bắt đầu.")
        frame_count = 0
        last_log_time = time.time()

        while True:
            try:
                # Nhận gói dữ liệu 4 phần tử từ Putter
                rgb_frame, depth_frame, fgmask, _ = await frame_queue.get()
                
                # Cập nhật và log FPS của Vòng Lặp Nhanh
                frame_count += 1
                current_time = time.time()
                if (current_time - last_log_time) > FAST_LOOP_LOG_INTERVAL:
                    fps = frame_count / (current_time - last_log_time)
                    logger.info(f"🚀 Fast Loop FPS: {fps:.2f}")
                    frame_count = 0
                    last_log_time = current_time

                # Tối ưu: Áp dụng mặt nạ trừ nền trước khi chạy YOLO
                rgb_foreground = cv2.bitwise_and(rgb_frame, rgb_frame, mask=fgmask)

                # Chạy detection trên ảnh đã được trừ nền
                detection_results = await asyncio.to_thread(self.detector.run_detection, rgb_foreground)
                
                # Chuẩn bị danh sách detection cho tracker
                detections = []
                if detection_results:
                    keypoints_data, boxes_data = detection_results
                    for i in range(len(boxes_data)):
                        detections.append({
                            "bbox": boxes_data[i], "keypoints": keypoints_data[i],
                            "tof_depth_map": depth_frame
                        })

                # Ghép cặp với tracker
                matches, unmatched_indices = self.tracker.match(detections)
                
                # Gửi tác vụ cho các detection mới
                for idx in unmatched_indices:
                    new_detection = detections[idx]
                    new_track_id = self.tracker.register_new_track(new_detection)
                    
                    if new_track_id is not None:
                        task_packet = {
                            "track_id": new_track_id,
                            "detection_data": new_detection,
                            "rgb_frame": rgb_frame
                        }
                        await self.reid_task_queue.put(task_packet)

            except Exception as e:
                logger.error(f"Lỗi trong Vòng lặp nhanh: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def run_slow_worker(self, worker_id: int, final_result_queue: asyncio.Queue):
        """Worker chậm: Lấy tác vụ, thực hiện xử lý sâu và gửi kết quả."""
        logger.info(f"🐌 Worker xử lý sâu #{worker_id} đã bắt đầu.")
        while True:
            try:
                task_packet = await self.reid_task_queue.get()
                
                async with self.slow_task_semaphore:
                    logger.info(f"🐌 Worker #{worker_id} đang xử lý sâu cho track_id: {task_packet['track_id']}...")
                    processed_data = await self._process_single_detection(task_packet)
                    
                    if processed_data:
                        await final_result_queue.put(processed_data)
                        logger.info(f"✅ Worker #{worker_id} hoàn tất xử lý cho track_id: {task_packet['track_id']}")

                self.reid_task_queue.task_done()
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong Worker #{worker_id}: {e}", exc_info=True)

    async def _process_single_detection(self, task_packet: Dict[str, Any]) -> Dict[str, Any] | None:
        """Hàm logic để xử lý sâu một detection, tách ra để dễ đọc và quản lý."""
        track_id = task_packet["track_id"]
        detection = task_packet["detection_data"]
        rgb_frame = task_packet["rgb_frame"]
        bbox, keypoints, tof_depth_map = detection["bbox"], detection["keypoints"], detection["tof_depth_map"]

        human_crop = rgb_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if human_crop.size == 0: return None

        try:
            # <<< GIẢI PHÁP: Gọi đúng phương thức của PoseColorSignatureExtractor >>>
            # Vui lòng xác nhận tên phương thức đúng. Tôi giả định là 'get_color_signature'.
            tasks_to_run = [
                asyncio.to_thread(self.height_estimator.estimate, keypoints, tof_depth_map),
                asyncio.to_thread(self.pose_processor.get_color_signature, human_crop, keypoints),
                asyncio.to_thread(extract_body_parts_from_frame, human_crop, keypoints)
            ]
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            if any(isinstance(res, Exception) for res in results):
                logger.error(f"Tác vụ con cho track_id {track_id} thất bại. Errors: {results}")
                return None
            
            (height_m, height_status), (body_color, _), body_parts = results

            return {
                "track_id": track_id, "bbox": bbox, "keypoints": keypoints,
                "human_box": human_crop, "body_color": body_color, "body_parts": body_parts,
                "est_height_m": height_m, "height_status": height_status,
                "camera_id": self.config.get("device_id", "opi_01")
            }
        except Exception as e:
            logger.error(f"Lỗi khi xử lý detection cho track_id {track_id}: {e}", exc_info=True)
            return None


async def start_processor(
    frame_queue: LatestFrameQueue,
    final_result_queue: asyncio.Queue,
    tracker: Track3DPro,
    config: dict
):
    """Khởi tạo và chạy FrameProcessor với kiến trúc hai tầng."""
    logger.info("Khởi động Processor (Kiến trúc hai tầng)...")
    
    processor = FrameProcessorFinal(config=config, tracker=tracker)
    all_processor_tasks = []
    try:
        fast_loop_task = asyncio.create_task(processor.run_fast_loop(frame_queue))
        slow_worker_tasks = [
            asyncio.create_task(processor.run_slow_worker(i, final_result_queue))
            for i in range(NUM_SLOW_WORKERS)
        ]
        all_processor_tasks = [fast_loop_task] + slow_worker_tasks
        await asyncio.gather(*all_processor_tasks)
    except asyncio.CancelledError:
        logger.info("Processor task bị hủy.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True)
    finally:
        for task in all_processor_tasks:
            if not task.done():
                task.cancel()