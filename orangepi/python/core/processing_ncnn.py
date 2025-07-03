

import asyncio
import cv2
import numpy as np

import random
# import orjson
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import os
import time

from utils.yolo_pose import HumanDetection
# from utils.yolo_pose import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor 
from utils.cut_body_part import extract_body_parts_from_frame

from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)


# Tạo thread pool với số lượng worker phù hợp
thread_pool = ThreadPoolExecutor()

class FrameProcessor:
    def __init__(self):
        self.detector = HumanDetection()
        self.pose_processor = PoseColorSignatureExtractor()
        self.fps_avg = 0.0
        self.call_count = 0
        logger.info("Start FrameProcessor successfully")

    async def run_detection_async(self, frame) -> Tuple[np.ndarray, List]:
        """
        Chạy detection bất đồng bộ
        """
        return await asyncio.to_thread(self.detector.run_detection, frame)

    async def process_body_color_async(self, frame, keypoints):
        """
        Xử lý body color bất đồng bộ
        """
        return await self.pose_processor.process_body_color_async(frame, keypoints, True)

    async def process_body_parts_async(self, frame, keypoints):
        """
        Xử lý body parts bất đồng bộ
        """
        return await asyncio.to_thread(extract_body_parts_from_frame, frame, keypoints)
    
    async def process_human_async(self, human_box, map_keypoints, frame_id):
        body_parts = await self.process_body_parts_async(human_box, map_keypoints)
        body_color = await self.process_body_color_async(human_box, map_keypoints)
        return {
            "uuid": frame_id,
            "human_box": human_box,
            "body_parts": body_parts,
            "body_color": body_color,
        }

    async def process_frame_queue(self, frame_queue: asyncio.Queue, proccessing_queue: asyncio.Queue):
        """
        Xử lý frame queue với tất cả các tác vụ nặng đều được thực hiện bất đồng bộ
        """
        frame_number = 0

        while True:
            try:
                start_time = time.time()
                
                frame = await frame_queue.get()
                if frame is None:
                    raise ValueError("Frame is None")

                keypoints_data, boxes_data = await self.run_detection_async(frame)

                if not boxes_data:
                    logger.warning("No boxes detected in the current frame. Skipping processing.") 
                    frame_queue.task_done()
                    continue

                # Tạo tasks cho từng human
                tasks = []
                for index, (keypoints, box) in enumerate(zip(keypoints_data, boxes_data)):
                    frame_id = generate_uuid()
                    human_box = frame[box[1]:box[3], box[0]:box[2]]
                    map_keypoints = self.detector.transform_keypoints_to_local(box, keypoints)
                    task = asyncio.create_task(self.process_human_async(human_box, map_keypoints, frame_id))
                    tasks.append(task)

                # Chờ tất cả tasks hoàn thành
                results = await asyncio.gather(*tasks)

                # Đưa kết quả vào queue
                for result in results:
                    if result["human_box"] is not None and result["human_box"].size > 0:
                        await proccessing_queue.put(result)

                frame_number += 1
                end_time = time.time()
                duration = end_time - start_time
                fps_current = 1 / duration if duration > 0 else 0
                self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
                self.call_count += 1
                logger.info(f"FPS processing: {self.fps_avg:.2f}")

                frame_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Processing task cancelled.")
                break

            except ValueError as ve:
                if str(ve) == "Frame is None":
                    logger.error("Frame is None, skipping processing.")
                else:
                    logger.error(f"ValueError occurred: {ve}")
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

async def serialize_feature_data(keypoints_data=None, boxes_data=None, body_color_data=None, uuid=None):
    """
    Serializes feature data into a compact byte format using orjson.
    
    Parameters:
        keypoints_data (np.ndarray or list, optional): Keypoint coordinates.
        boxes_data (list, optional): Bounding box information.
        body_color_data (list, optional): List of color signature data.
        uuid (str, optional): Unique identifier.
    
    Returns:
        bytes: Serialized feature data.
    """
    try:
        def convert_to_bytes():
            feature_dict = {}
            
            if keypoints_data is not None:
                keypoints_data_list = keypoints_data.tolist() if isinstance(keypoints_data, np.ndarray) else keypoints_data
                feature_dict["keypoints_data"] = keypoints_data_list
            
            if boxes_data is not None:
                feature_dict["boxes_data"] = boxes_data
            
            if uuid is not None:
                feature_dict["uuid"] = uuid
            
            if body_color_data is not None:
                body_color_data_list = [
                    [idx, color.tolist() if isinstance(color, np.ndarray) else color] 
                    for idx, color in enumerate(body_color_data) if color is not None
                ]
                feature_dict["color_signature_data"] = body_color_data_list
                # logger.info(f"body_color_data_list: {body_color_data_list}")
            
            return orjson.dumps(feature_dict)
        
        return await asyncio.get_event_loop().run_in_executor(None, convert_to_bytes)
    
    except Exception as e:
        logger.error(f"Failed to serialize feature data: {e}")
        return b""

def deserialize_feature_data(feature_dict):
    """
    Deserializes feature data from a dictionary format back to structured data.
    
    Parameters:
        feature_dict (dict): Dictionary containing serialized feature data.
    
    Returns:
        list: Reconstructed body_color_data list with np.ndarray values and None placeholders.
    """
    try:
        max_index = max(idx for idx, _ in feature_dict.get("color_signature_data", [])) if feature_dict.get("color_signature_data") else -1
        body_color_data = [None] * (max_index + 1)
        
        for idx, color in feature_dict.get("color_signature_data", []):
            body_color_data[idx] = np.array(color, dtype=np.uint8)
        
        return body_color_data
    except Exception as e:
        logger.error(f"Failed to deserialize feature data: {e}")
        return []

def generate_uuid():
    """
    Tạo chuỗi UUID ngẫu nhiên 6 chữ số.
    """
    return f"{random.randint(100000, 999999):06}"

async def start_processor(frame_queue: asyncio.Queue, proccessing_queue: asyncio.Queue):
    """
    Khởi động một processor worker để xử lý song song
    
    Args:
        frame_queue: Queue chứa các frame đầu vào
        human_queue: Queue chứa các thông tin từ con người
        head_queue: Queue chứa các thông tin từ đầu
        right_arm_queue: Queue chứa các thông tin từ cánh tay phải
        left_arm_queue: Queue chứa các thông tin từ cánh tay trái
    """
    logger.info("Starting processor worker...")
    
    # Khởi tạo một processor worker
    processor = FrameProcessor()
    
    # Tạo task cho processor
    task = asyncio.create_task(
        processor.process_frame_queue(
            frame_queue=frame_queue,
            proccessing_queue=proccessing_queue
        )
    )
    
    try:
        # Chạy task processor
        await task
    except asyncio.CancelledError:
        logger.info("Processor task is being cancelled...")
        task.cancel()
        await task  # Chờ task kết thúc
        logger.info("Processor task has been cancelled.")
    finally:
        thread_pool.shutdown(wait=True)

# Sửa lại hàm main để test
async def main():
    frame_queue = asyncio.Queue()
    processed_frame_queue = asyncio.Queue()
    
    try:
        await start_processor(frame_queue, processed_frame_queue)
    except Exception as e:
        logger.error(f"Main processing error: {e}")

if __name__ == "__main__":
    asyncio.run(main())