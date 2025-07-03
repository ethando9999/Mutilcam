import asyncio
import os
import numpy as np
import cv2 
from utils.logging_python_orangepi import get_logger


loggger = get_logger(__name__)

async def save_frame(frame, output_dir, uuid, box_type):
    """Lưu frame vào thư mục tương ứng."""
    if frame is not None and isinstance(frame, np.ndarray):
        uuid_dir = os.path.join(output_dir, f"{uuid}")
        os.makedirs(uuid_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        file_path = os.path.join(uuid_dir, f"{box_type}.jpg")
        cv2.imwrite(file_path, frame)
        loggger.info(f"Saved: {file_path}")

async def start_processing(human_queue: asyncio.Queue, head_queue: asyncio.Queue, right_arm_queue: asyncio.Queue, left_arm_queue: asyncio.Queue):
    output_dir = "frame_dir"
    os.makedirs(output_dir, exist_ok=True)

    loggger.info("Start processing...")

    while True:
        # Lấy frame từ các hàng đợi
        human_queue_dict = await human_queue.get()
        head_queue_dict = await head_queue.get()
        right_arm_queue_dict = await right_arm_queue.get()
        left_arm_queue_dict = await left_arm_queue.get()

        if (human_queue_dict is None and head_queue_dict is None and 
            right_arm_queue_dict is None and left_arm_queue_dict is None):
            break  # Thoát nếu nhận tín hiệu dừng

        # Tạo danh sách các task để lưu frame
        tasks = []

        # Lưu frame từ human_queue
        if human_queue_dict and human_queue_dict.get("frame") is not None:
            tasks.append(save_frame(human_queue_dict.get("frame"), output_dir, human_queue_dict.get("uuid"), "human"))
        else:
            loggger.warning("No human frame to save.")

        # Lưu frame từ head_queue
        if head_queue_dict and head_queue_dict.get("frame") is not None:
            tasks.append(save_frame(head_queue_dict.get("frame"), output_dir, head_queue_dict.get("uuid"), "head"))
        else:
            loggger.warning("No head frame to save.")

        # Lưu frame từ right_arm_queue
        if right_arm_queue_dict and right_arm_queue_dict.get("frame") is not None:
            tasks.append(save_frame(right_arm_queue_dict.get("frame"), output_dir, right_arm_queue_dict.get("uuid"), "right_arm"))
        else:
            loggger.warning("No right arm frame to save.")

        # Lưu frame từ left_arm_queue
        if left_arm_queue_dict and left_arm_queue_dict.get("frame") is not None:
            tasks.append(save_frame(left_arm_queue_dict.get("frame"), output_dir, left_arm_queue_dict.get("uuid"), "left_arm"))
        else:
            loggger.warning("No left arm frame to save.")

        # Chạy tất cả các task lưu frame đồng thời
        await asyncio.gather(*tasks)

        # Đánh dấu hàng đợi đã hoàn thành
        human_queue.task_done()
        head_queue.task_done()
        right_arm_queue.task_done()
        left_arm_queue.task_done()