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
from utils.pose_color_signature import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame

logger = get_logger(__name__)

# Tạo thread pool với số lượng worker phù hợp
thread_pool = ThreadPoolExecutor()

class FrameProcessor:
    def __init__(self):
        self.detector = HumanDetection()
        self.pose_processor = PoseColorSignatureExtractor()

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
    
    async def process_body_parts_async(self, frame, keypoints):
        """
        Xử lý body parts bất đồng bộ
        """
        def body_parts_task():
            return extract_body_parts_from_frame(frame, keypoints)

        return await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            body_parts_task
        )

    async def process_frame_queue2(self, frame_queue: asyncio.Queue, processed_frame_queue: asyncio.Queue):
        """
        Xử lý frame queue với tất cả các tác vụ nặng đều được thực hiện bất đồng bộ
        """
        # Tạo thư mục human_detection nếu chưa tồn tại
        os.makedirs("human_detection", exist_ok=True)
        frame_number = 0

        while True:
            try:
                # Lấy frame từ queue 
                frame = await frame_queue.get()

                if frame is None:
                    raise ValueError("Failed to decode bytes to a valid image.")
                
                frame_number += 1

                # Chạy detection bất đồng bộ
                keypoints_data, boxes_data = await self.run_detection_async(frame)

                # Kiểm tra nếu boxes_data rỗng
                if not boxes_data:
                    logger.warning("No boxes detected in the current frame. Skipping processing.")
                    frame_queue.task_done()  # Đánh dấu khung đã hoàn thành
                    continue  # Bỏ qua frame này   

                # Tạo thư mục cho frame hiện tại
                frame_path = f"human_detection/frame_{frame_number}"
                os.makedirs(frame_path, exist_ok=True)

                # Vẽ hộp và các điểm trên ảnh
                annotated_img = self.detector.draw_boxes_and_edges()  # Thêm tham số vào hàm

                # Lưu annotated_img vào thư mục human_detection
                output_path = f"{frame_path}/annotated_image.jpg"
                cv2.imwrite(output_path, annotated_img)  # Lưu ảnh đã được annotate 
                        

                # Xử lý body color bất đồng bộ cho tất cả keypoints
                for index, (keypoints, box) in enumerate(zip(keypoints_data, boxes_data)):

                    frame_id = generate_uuid()  # Tạo UUID cho mỗi human box

                    # map keypoints từ ảnh gốc sang tọa độ trong bounding box.
                    map_keypoints = self.detector.transform_keypoints_to_local(box, keypoints)

                    # Cắt human_box dựa trên boxes
                    human_box = frame[box[1]:box[3], box[0]:box[2]]  # boxes: (x1, y1, x2, y2)       

                    # Cắt body part 
                    body_parts = await self.process_body_parts_async(human_box, map_keypoints)

                    # Tạo thư mục cho từng người
                    human_path = f"{frame_path}/human{index + 1}"
                    os.makedirs(human_path, exist_ok=True)

                    # Lưu human_box
                    if human_box is not None and human_box.size > 0:
                        human_box_path = f"{human_path}/human{index + 1}.jpg" 
                        cv2.imwrite(human_box_path, human_box)
                    else:
                        logger.error(f"Human box is empty for index {index}.")

                    # Lưu body parts nếu không rỗng
                    if body_parts["head"] is not None and body_parts["head"].size > 0:
                        head_path = f"{human_path}/head.jpg"
                        cv2.imwrite(head_path, body_parts["head"])
                    else:
                        logger.error(f"Head part is empty for human {index + 1}.")

                    if body_parts["right_arm"] is not None and body_parts["right_arm"].size > 0:
                        right_arm_path = f"{human_path}/right_arm.jpg"
                        cv2.imwrite(right_arm_path, body_parts["right_arm"])
                    else:
                        logger.error(f"Right arm part is empty for human {index + 1}.")

                    if body_parts["left_arm"] is not None and body_parts["left_arm"].size > 0:    
                        left_arm_path = f"{human_path}/left_arm.jpg"
                        cv2.imwrite(left_arm_path, body_parts["left_arm"])
                    else:
                        logger.error(f"Left arm part is empty for human {index + 1}.")
                        
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

    async def process_frame_queue(self, frame_queue: asyncio.Queue, human_queue: asyncio.Queue, head_queue: asyncio.Queue, right_arm_queue: asyncio.Queue, left_arm_queue: asyncio.Queue):
        """
        Xử lý frame queue với tất cả các tác vụ nặng đều được thực hiện bất đồng bộ
        """
        frame_number = 0
        start_time = asyncio.get_event_loop().time()  # Thêm biến để theo dõi thời gian bắt đầu
        fps = 0  # Biến để lưu trữ FPS

        while True:
            try:
                # Lấy frame từ queue 
                frame = await frame_queue.get()

                if frame is None:
                    raise ValueError("Failed to decode bytes to a valid image.")                

                # Chạy detection bất đồng bộ
                keypoints_data, boxes_data = await self.run_detection_async(frame)

                # Kiểm tra nếu boxes_data rỗng
                if not boxes_data:
                    logger.warning("No boxes detected in the current frame. Skipping processing.")
                    frame_queue.task_done()  # Đánh dấu khung đã hoàn thành
                    continue  # Bỏ qua frame này   

                # Xử lý body color bất đồng bộ cho tất cả keypoints
                for index, (keypoints, box) in enumerate(zip(keypoints_data, boxes_data)):

                    frame_id = generate_uuid()  # Tạo UUID cho mỗi human box

                    # map keypoints từ ảnh gốc sang tọa độ trong bounding box.
                    map_keypoints = self.detector.transform_keypoints_to_local(box, keypoints)

                    # Cắt human_box dựa trên boxes
                    human_box = frame[box[1]:box[3], box[0]:box[2]]  # boxes: (x1, y1, x2, y2)       

                    # Cắt body part 
                    body_parts = await self.process_body_parts_async(human_box, map_keypoints)

                    # Tính màu cho body color
                    body_color_data = await self.process_body_color_async(human_box, map_keypoints)

                    # Ghi lại loại dữ liệu và giá trị của body_color_data
                    # logger.info(f"body_color_data type: {type(body_color_data)}, value: {body_color_data}")

                    # Tạo feature_data gồm body_color_data + uuid 
                    color_uuid_data = await serialize_feature_data(body_color_data=body_color_data, uuid=frame_id)

                    # Tạo feature_data chỉ uuid 
                    uuid_data = await serialize_feature_data(uuid=frame_id)

                    # Nén image to bytes
                    if human_box is not None and human_box.size > 0:
                        human_bytes = await compress_frame(human_box, quality=50) 
                        await human_queue.put({
                            "frame": human_bytes,
                            "uuid": frame_id,
                            "feature": color_uuid_data,
                        })

                    if body_parts["head"] is not None and body_parts["head"].size > 0:
                        head_bytes = await compress_frame(body_parts["head"], quality=100)
                        await head_queue.put({
                            "frame": head_bytes,
                            "uuid": frame_id,
                            "feature": uuid_data, 
                        })
                        
                    if body_parts["right_arm"] is not None and body_parts["right_arm"].size > 0:
                        right_arm_bytes = await compress_frame(body_parts["right_arm"], quality=100)
                        await right_arm_queue.put({
                            "frame": right_arm_bytes,
                            "uuid": frame_id,
                            "feature": uuid_data,
                        })

                    if body_parts["left_arm"] is not None and body_parts["left_arm"].size > 0:    
                        left_arm_bytes = await compress_frame(body_parts["left_arm"], quality=100)
                        await left_arm_queue.put({
                            "frame": left_arm_bytes,
                            "uuid": frame_id,
                            "feature": uuid_data, 
                        })  

                frame_number += 1
                # Tính toán FPS mỗi giây
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1:
                    fps = frame_number / elapsed_time
                    logger.info(f"FPS processing: {fps:.2f}")
                    start_time = current_time
                    frame_number = 0  # Đặt lại số khung hình

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
    
    async def process_body_parts_async(self, frame, keypoints):
        """
        Xử lý body parts bất đồng bộ
        """
        def body_parts_task():
            return extract_body_parts_from_frame(frame, keypoints)

        return await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            body_parts_task
        )

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

async def start_processor(frame_queue: asyncio.Queue, human_queue: asyncio.Queue, head_queue: asyncio.Queue, right_arm_queue: asyncio.Queue, left_arm_queue: asyncio.Queue):
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
            human_queue=human_queue,
            head_queue=head_queue,
            right_arm_queue=right_arm_queue,
            left_arm_queue=left_arm_queue
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