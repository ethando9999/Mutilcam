import asyncio
import cv2
import numpy as np
from utils.camera_log import get_logger
import random
import orjson
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

from utils.yolo_pose import HumanDetection
from utils.pose_cluster import PoseClusterProcessor

logger = get_logger(__name__)
# Tạo thread pool với số lượng worker phù hợp
thread_pool = ThreadPoolExecutor()

class FrameProcessor:
    def __init__(self):
        self.detector = HumanDetection()
        self.pose_processor = PoseClusterProcessor()

    async def run_detection_async(self, frame) -> Tuple[np.ndarray, List]:
        """
        Chạy detection bất đồng bộ
        """
        def detection_task():
            return self.detector.run_detection(frame)

        return await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            detection_task
        )

    async def process_body_color_async(self, frame, keypoints):
        """
        Xử lý body color bất đồng bộ
        """
        def color_task():
            return self.pose_processor.process_body_color(frame, keypoints)

        return await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            color_task
        )

    async def process_frame_queue(self, frame_queue: asyncio.Queue, processed_frame_queue: asyncio.Queue):
        """
        Xử lý frame queue với tất cả các tác vụ nặng đều được thực hiện bất đồng bộ
        """
        while True:
            try:
                # Lấy frame từ queue
                frame = await frame_queue.get()
                frame_id = generate_uuid()

                if frame is None:
                    raise ValueError("Failed to decode bytes to a valid image.")

                # Chạy detection bất đồng bộ
                keypoints_data, boxes_data = await self.run_detection_async(frame)

                # Xử lý body color bất đồng bộ cho tất cả keypoints
                body_color_tasks = [
                    self.process_body_color_async(frame, keypoints)
                    for keypoints in keypoints_data
                ]
                # Chạy tất cả các task xử lý màu đồng thời
                body_color_data = await asyncio.gather(*body_color_tasks)

                # Serialize feature data bất đồng bộ
                feature_data_bytes = await serialize_feature_data(
                    keypoints_data,
                    boxes_data,
                    body_color_data
                )

                frame_bytes = await compress_frame(frame=frame, quality=80)

                # Đẩy kết quả vào processed queue 
                await processed_frame_queue.put({
                    "frame": frame_bytes,
                    "uuid": frame_id,
                    "feature_data": feature_data_bytes
                })
                
                frame_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Processing task cancelled.")
                break

            except ValueError as ve:
                logger.error(f"Frame decoding error: {ve}")
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error during frame processing: {e}")
                await asyncio.sleep(1)

# Các hàm hỗ trợ giữ nguyên như cũ
async def decode_frame(frame_bytes: bytes):
    def decode():
        np_array = np.frombuffer(frame_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return await asyncio.get_event_loop().run_in_executor(
        thread_pool,
        decode
    )

async def compress_frame(frame: np.ndarray, quality: int = 80) -> bytes:
    """
    Nén frame thành định dạng JPEG.
    
    Args:
        frame: Frame cần nén (numpy array)
        quality: Chất lượng nén JPEG (1-100)
    
    Returns:
        bytes: Dữ liệu đã được nén
    """
    try:
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded_frame.tobytes()
    except Exception as e:
        logger.error(f"Lỗi khi nén frame: {e}")
        return None

async def serialize_feature_data(keypoints_data, boxes_data, body_color_data):
    try:
        def convert_to_bytes():
            keypoints_data_list = keypoints_data.tolist() if isinstance(keypoints_data, np.ndarray) else keypoints_data
            body_color_data_list = [
                [color.tolist() if isinstance(color, np.ndarray) else color for color in person_data]
                for person_data in body_color_data
            ]
            
            feature_dict = {
                "keypoints_data": keypoints_data_list,
                "boxes_data": boxes_data,
                "color_signature_data": body_color_data_list
            }
            
            return orjson.dumps(feature_dict)

        return await asyncio.get_event_loop().run_in_executor(
            thread_pool, 
            convert_to_bytes
        )
    
    except Exception as e:
        logger.error(f"Failed to serialize feature data: {e}")
        return b""

def generate_uuid():
    """
    Tạo chuỗi UUID ngẫu nhiên 6 chữ số.
    """
    return f"{random.randint(100000, 999999):06}"

async def start_processor(frame_queue: asyncio.Queue, processed_frame_queue: asyncio.Queue, num_workers: int = 1):
    """
    Khởi động nhiều processor workers để xử lý song song
    
    Args:
        frame_queue: Queue chứa các frame đầu vào
        processed_frame_queue: Queue chứa các frame đã xử lý
        num_workers: Số lượng worker processor chạy song song
    """
    logger.info(f"Starting {num_workers} processor workers...")
    
    # Khởi tạo các processor workers
    processors = [FrameProcessor() for _ in range(num_workers)]
    
    # Tạo các tasks cho từng processor
    processor_tasks = []
    for processor in processors:
        task = asyncio.create_task(
            processor.process_frame_queue(
                frame_queue=frame_queue,
                processed_frame_queue=processed_frame_queue
            )
        )
        processor_tasks.append(task)
    
    try:
        # Chạy tất cả các processor tasks
        await asyncio.gather(*processor_tasks)
    except asyncio.CancelledError:
        logger.info("Processor tasks are being cancelled...")
        # Hủy tất cả các tasks
        for task in processor_tasks:
            task.cancel()
        # Chờ các tasks kết thúc
        await asyncio.gather(*processor_tasks, return_exceptions=True)
        logger.info("All processor tasks have been cancelled.")
    finally:
        thread_pool.shutdown(wait=True)

# Sửa lại hàm main để test
async def main():
    frame_queue = asyncio.Queue()
    processed_frame_queue = asyncio.Queue()
    
    try:
        await start_processor(frame_queue, processed_frame_queue, num_workers=2)
    except Exception as e:
        logger.error(f"Main processing error: {e}")

if __name__ == "__main__":
    asyncio.run(main())