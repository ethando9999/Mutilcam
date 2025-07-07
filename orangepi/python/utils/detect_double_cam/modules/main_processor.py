# modules/main_processor.py (phiên bản mới)
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from utils.logging_python_orangepi import get_logger

# <<<<<<<<<<<< THAY ĐỔI IMPORT >>>>>>>>>>>>
from utils.detect_double_cam.main_detect import MainDetect

logger = get_logger(__name__)

class MainProcessor:
    def __init__(self, calib_file_path, results_dir):
        """Khởi tạo MainProcessor."""
        # Tạo một instance duy nhất của MainDetect
        self.main_detector = MainDetect(calib_file_path, results_dir)
        # Thread pool để chạy hàm xử lý nặng
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

    async def run(self, stereo_frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
        """
        Vòng lặp chính, lấy cặp ảnh stereo từ queue và xử lý.
        """
        loop = asyncio.get_event_loop()
        logger.info("Main Processor started. Waiting for synchronized stereo frames...")
        
        while True:
            # Chờ đợi một cặp frame hoàn chỉnh
            # Giả định producer đẩy một tuple (rgb_frame, depth_frame, amp_frame)
            rgb_frame, tof_depth, tof_amp = await stereo_frame_queue.get()
            
            # Gửi cặp frame đi xử lý trong một thread khác để không block event loop
            future = loop.run_in_executor(
                self.executor, self.main_detector.process_stereo_pair, rgb_frame, tof_depth, tof_amp
            )
            
            # Lấy kết quả khi xử lý xong
            valid_persons_data, annotated_rgb_frame = await future
            
            # Đẩy kết quả đã xử lý (danh sách người hợp lệ) vào hàng đợi tiếp theo
            if valid_persons_data:
                await processing_queue.put(valid_persons_data)
                
            stereo_frame_queue.task_done()

    def close(self):
        self.executor.shutdown(wait=True)
        self.main_detector.release()

async def start_main_processor(stereo_frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, calib_path: str, results_dir: str):
    processor = MainProcessor(calib_path, results_dir)
    try:
        await processor.run(stereo_frame_queue, processing_queue)
    except asyncio.CancelledError:
        logger.info("Main Processor task cancelled.")
    finally:
        processor.close()