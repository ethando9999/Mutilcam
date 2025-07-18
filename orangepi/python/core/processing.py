import asyncio
import cv2
import numpy as np
import orjson
from utils.logging_python_orangepi import get_logger
import random
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from utils.yolo_pose import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame
import os 
from datetime import datetime

logger = get_logger(__name__)

# Thread pool với số lượng worker giới hạn để tối ưu cho thiết bị biên
thread_pool = ThreadPoolExecutor(max_workers=4)

class FrameProcessor:
    def __init__(self, batch_size=2): 
        self.detector = HumanDetection()
        self.pose_processor = PoseColorSignatureExtractor()
        self.fps_avg = 0.0
        self.call_count = 0
        self.semaphore = asyncio.Semaphore(4) 
        self.batch_size = batch_size

    async def run_detection_async(self, frame) -> Tuple[np.ndarray, List]:
        """Chạy phát hiện bất đồng bộ."""
        start_time = time.time()
        result = await asyncio.to_thread(self.detector.run_detection, frame)
        logger.debug(f"Phát hiện mất {time.time() - start_time:.2f} giây")
        return result

    async def process_body_color_async(self, frame, keypoints):
        """Xử lý màu sắc cơ thể bất đồng bộ."""
        start_time = time.time()
        result = await self.pose_processor.process_body_color_async(frame, keypoints, True)
        logger.debug(f"Xử lý màu sắc cơ thể mất {time.time() - start_time:.2f} giây")
        return result

    async def process_body_parts_async(self, frame, keypoints):
        """Xử lý các bộ phận cơ thể bất đồng bộ."""
        start_time = time.time()
        result = await asyncio.to_thread(extract_body_parts_from_frame, frame, keypoints)
        logger.debug(f"Xử lý các bộ phận cơ thể mất {time.time() - start_time:.2f} giây")
        return result
    
    async def process_human_async(self, human_box, map_keypoints, frame_id, box):
        """Xử lý một người bất đồng bộ với semaphore."""
        async with self.semaphore:
            try:
                body_parts = await self.process_body_parts_async(human_box, map_keypoints)
                # body_color = await self.process_body_color_async(human_box, map_keypoints)
                return {
                    "frame_id": frame_id,
                    "human_box": human_box,
                    "body_parts": body_parts,
                    "body_color": None,
                    "bbox": box,
                    "map_keypoints": map_keypoints,
                    "head_point_3d": None, 
                    "distance_mm": None, 
                    "time_detect": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Lỗi xử lý người cho frame {frame_id}: {e}", exc_info=True)
                return None

    async def process_frame_queue(self, frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
        """Xử lý khung hình từ hàng đợi theo lô, không dừng vĩnh viễn khi gặp None.""" 
        frame_number = 0

        while True:
            start_time = time.time()
            logger.debug(f"Kích thước hàng đợi trước lô: {frame_queue.qsize()}")

            batch_frames = []

            # Bước 1: chờ blocking 1 item để không busy-loop liên tục khi queue rỗng
            try:
                item = await frame_queue.get()
            except asyncio.CancelledError:
                logger.info("process_frame_queue bị hủy từ bên ngoài.")
                break 
            try:
                # Xử lý item đầu tiên
                if item is None:
                    # Nếu nhận None, chỉ log và bỏ qua
                    logger.debug("Nhận được item None, bỏ qua và tiếp tục chờ frame khác...")
                else:
                    cam_idx, frame_path = item
                    if not os.path.exists(frame_path):
                        logger.debug(f"Đường dẫn khung hình {frame_path} không tồn tại, bỏ qua.")
                    else:
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            logger.error(f"Không thể đọc khung hình từ {frame_path}, bỏ qua.")
                        else:
                            batch_frames.append((cam_idx, frame))
                            logger.debug(f"Lấy khung hình đầu tiên từ camera {cam_idx}")
            except Exception as e:
                logger.error(f"Lỗi khi lấy item đầu tiên: {e}", exc_info=True)
            finally:
                frame_queue.task_done()

            # Bước 2: cố gắng lấp đầy batch thêm (non-blocking) cho đến batch_size
            for _ in range(self.batch_size - 1):
                try:
                    item = frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    if item is None:
                        logger.debug("Nhận được item None trong fill batch, bỏ qua.")
                        # chỉ bỏ qua, không dừng
                    else:
                        cam_idx, frame_path = item
                        if not os.path.exists(frame_path):
                            logger.debug(f"Đường dẫn khung hình {frame_path} không tồn tại, bỏ qua.")
                        else:
                            frame = cv2.imread(frame_path)
                            if frame is None:
                                logger.error(f"Không thể đọc khung hình từ {frame_path}, bỏ qua.")
                            else:
                                batch_frames.append((cam_idx, frame))
                                logger.debug(f"Lấy thêm khung hình từ camera {cam_idx}")
                except Exception as e:
                    logger.error(f"Lỗi khi fill batch: {e}", exc_info=True)
                finally:
                    frame_queue.task_done()

            # Nếu không có frame nào hợp lệ trong batch, chờ chút rồi tiếp tục vòng
            if not batch_frames:
                await asyncio.sleep(0.01)
                continue

            # Bước 3: xử lý batch_frames
            try:
                for i, (cam_idx, frame) in enumerate(batch_frames):
                    current_frame_number = frame_number + i

                    # run_detection_async: trả về keypoints_data và boxes_data
                    keypoints_data, boxes_data = await self.run_detection_async(frame)
                    
                    if not boxes_data:
                        logger.warning(f"Không phát hiện hộp nào trong khung hình từ camera {cam_idx}. Bỏ qua xử lý.")
                        continue

                    tasks = []
                    for index, (keypoints, box) in enumerate(zip(keypoints_data, boxes_data)):
                        # frame_id = generate_uuid()
                        human_box = frame[box[1]:box[3], box[0]:box[2]]
                        map_keypoints = self.detector.transform_keypoints_to_local(box, keypoints)
                        task = asyncio.create_task(self.process_human_async(human_box, map_keypoints, current_frame_number, box))
                        tasks.append(task)

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Lỗi trong xử lý người: {result}", exc_info=True)
                            continue
                        # Kiểm tra result["human_box"] tồn tại
                        if result and result.get("human_box") is not None and hasattr(result["human_box"], 'size') and result["human_box"].size > 0:
                            result["camera_id"] = cam_idx
                            await processing_queue.put(result)

                # Cập nhật FPS trung bình và log
                frame_number += len(batch_frames)
                end_time = time.time()
                duration = end_time - start_time
                fps_current = len(batch_frames) / duration if duration > 0 else 0
                # Giả sử self.fps_avg, self.call_count đã khởi tạo trước đó
                self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
                self.call_count += 1
                if frame_number % 10 == 0:
                    logger.info(f"FPS xử lý: {self.fps_avg:.2f}, Kích thước hàng đợi: {frame_queue.qsize()}")
            except asyncio.CancelledError:
                logger.info("Tác vụ xử lý batch bị hủy.")
                break
            except Exception as e:
                logger.error(f"Lỗi trong xử lý lô: {e}", exc_info=True)
                # Chờ chút trước khi tiếp tục để tránh loop nhanh khi lỗi liên tục
                await asyncio.sleep(1)

        # Khi ra khỏi vòng while (bị hủy), có thể log hoặc dọn dẹp thêm nếu cần
        logger.info("process_frame_queue kết thúc.")


def generate_uuid():
    """Tạo UUID ngẫu nhiên 6 chữ số."""
    return f"{random.randint(100000, 999999):06}"

async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
    """Khởi động worker xử lý song song."""
    logger.info("Khởi động worker xử lý...")
    processor = FrameProcessor()
    task = asyncio.create_task(processor.process_frame_queue(frame_queue, processing_queue))
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Tác vụ xử lý đang được hủy...")
        task.cancel()
        await task
        logger.info("Tác vụ xử lý đã được hủy.")
    finally:
        thread_pool.shutdown(wait=True)