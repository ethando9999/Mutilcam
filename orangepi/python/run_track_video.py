# file: python/run_track_from_video.py

import asyncio
import cv2
import numpy as np
import os
import shutil
import argparse
import time

# --- Import các module cốt lõi của dự án ---
# Đảm bảo các đường dẫn import này là chính xác so với vị trí của file
from core.processing_track import start_processor
from track_local.byte_track import TrackingManager
from utils.logging_python_orangepi import get_logger
import config

logger = get_logger(__name__)

# Thư mục tạm để lưu các frame được trích xuất từ video
TEMP_FRAMES_DIR = "temp_video_frames"

class VideoFrameProducer:
    """
    Một "Producer" giả lập, đọc các frame từ một file video,
    lưu chúng vào file tạm, và đưa đường dẫn vào hàng đợi.
    """
    def __init__(self, video_path: str, frame_queue: asyncio.Queue):
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.temp_dir = TEMP_FRAMES_DIR
        self.cap = None
        self.video_fps = 30.0 # Giá trị mặc định

    def setup(self):
        """Kiểm tra file video và chuẩn bị thư mục tạm."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Không tìm thấy file video tại: {self.video_path}")
        
        # Dọn dẹp thư mục tạm cũ nếu có và tạo mới
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Không thể mở file video: {self.video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Đã mở video '{self.video_path}' ({self.video_fps:.2f} FPS, {frame_count} frames).")

    async def run(self):
        """Vòng lặp chính: đọc frame, tạo dữ liệu giả và đưa vào hàng đợi."""
        logger.info(">>> VideoFrameProducer đã khởi động...")
        frame_idx = 0
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Đã xử lý hết video.")
                    break

                # Tạo một bản đồ độ sâu giả (mảng numpy toàn số 0)
                # Kích thước phải khớp với frame RGB
                height, width, _ = frame.shape
                fake_depth_map = np.zeros((height, width), dtype=np.uint16)

                # Tạo đường dẫn file tạm
                rgb_path = os.path.join(self.temp_dir, f"frame_{frame_idx:06d}.png")
                depth_path = os.path.join(self.temp_dir, f"frame_{frame_idx:06d}.npy")
                
                # Lưu frame và bản đồ độ sâu giả vào file
                # Chạy các tác vụ I/O này trong thread riêng để không block vòng lặp
                await asyncio.to_thread(cv2.imwrite, rgb_path, frame)
                await asyncio.to_thread(np.save, depth_path, fake_depth_map)
                
                # Đưa tuple đường dẫn vào hàng đợi, giống hệt như producer thật
                # Giả định amp_path là None
                await self.frame_queue.put((rgb_path, depth_path, None))
                
                # Điều khiển tốc độ xử lý để mô phỏng FPS của video gốc
                await asyncio.sleep(1.0 / self.video_fps)
                
                frame_idx += 1
        
        except Exception as e:
            logger.error(f"Lỗi trong VideoFrameProducer: {e}", exc_info=True)
        finally:
            logger.info("VideoFrameProducer đang dừng...")
            # Gửi tín hiệu kết thúc (None) tới các consumer
            await self.frame_queue.put(None)
            self.cleanup()

    def cleanup(self):
        """Dọn dẹp tài nguyên."""
        if self.cap:
            self.cap.release()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Đã xóa thư mục tạm: {self.temp_dir}")

async def main(video_path: str):
    logger.info("Khởi động ứng dụng chạy thực nghiệm từ video...")
    
    # 1. Tạo các hàng đợi (Queues)
    frame_queue = asyncio.Queue(maxsize=10)
    processing_queue = asyncio.Queue(maxsize=200) # Dành cho tracker
    socket_queue = asyncio.Queue(maxsize=1) # Dành cho sender
    people_count_queue = asyncio.Queue(maxsize=1)
    height_queue = asyncio.Queue(maxsize=1)

    # 2. Khởi tạo các thành phần chính
    producer = VideoFrameProducer(video_path=video_path, frame_queue=frame_queue)
    try:
        producer.setup()
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Lỗi khởi tạo producer: {e}")
        return

    # Khởi tạo TrackingManager
    tracker = TrackingManager(
        detection_queue=processing_queue,
        processed_queue=socket_queue, # Output của tracker sẽ vào đây
    )
    
    # 3. Tạo các tác vụ (Tasks)
    tasks = []
    try:
        producer_task = asyncio.create_task(producer.run())
        tasks.append(producer_task)

        # start_processor là hàm khởi tạo FrameProcessor và chạy nó
        processor_task = asyncio.create_task(start_processor(
            frame_queue, processing_queue, people_count_queue, height_queue,
            config.OPI_CONFIG['calib_file_path']
        ))
        tasks.append(processor_task)

        tracker_task = asyncio.create_task(tracker.run())
        tasks.append(tracker_task)

        logger.info(f"Đã khởi tạo {len(tasks)} tác vụ chính. Bắt đầu xử lý video.")
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong hàm main: {e}", exc_info=True)
    finally:
        logger.info("Đang dọn dẹp và đóng ứng dụng...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        producer.cleanup()
        logger.info("Hoàn tất.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline tracking từ file video.")
    parser.add_argument(
        '--video-path',
        type=str,
        default="remote_webcam_video.mp4",
        help="Đường dẫn tới file video cần xử lý."
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.video_path))
    except KeyboardInterrupt:
        logger.info("Đã nhận tín hiệu dừng từ người dùng (Ctrl+C).")