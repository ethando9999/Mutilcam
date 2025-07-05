import numpy as np
import asyncio
import cv2
import time
import os
import logging # Sử dụng logging chuẩn để ví dụ chạy được

# Giả lập các module của bạn để code có thể chạy
# Bạn hãy thay thế bằng các import gốc của mình
try:
    from utils.logging_python_orangepi import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    def get_logger(name):
        return logging.getLogger(name)

try:
    from .call_slave import CallSlave # Đổi tên từ call_slaver
except ImportError:
    # Giả lập lớp CallSlave để test
    class CallSlave:
        def __init__(self, *args, **kwargs): logging.info("Dummy CallSlave initialized.")
        def request_and_receive_tof_frames(self):
            time.sleep(0.05) # Giả lập độ trễ mạng
            depth = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)
            amp = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            return depth, amp
        def close(self): logging.info("Dummy CallSlave closed.")

try:
    from utils.rmbg_mog import BackgroundRemover
except ImportError:
    class BackgroundRemover:
        def remove_background(self, frame): return frame, frame, frame


logger = get_logger(__name__)

class FramePutter:
    def __init__(self, rgb_camera_id=0):
        self.stop_event = asyncio.Event()
        self.fps = 0
        self.prev_frame = None
        self.frame_count = 0
        self.start_time = None
        
        # Thư mục lưu trữ các loại frame
        self.rgb_frame_dir = "frames/rgb"
        self.depth_data_dir = "frames/depth"
        self.amp_frame_dir = "frames/amp"

        self.cam_index = 1 
        self.rgb_camera_id = rgb_camera_id
        
        # Khởi tạo các đối tượng xử lý
        self.background_remover = BackgroundRemover()
        # Khởi tạo CallSlave, nó sẽ cố gắng kết nối ngay lập tức
        # theo thiết kế phương pháp 3 (kết hợp)
        self.slave = CallSlave(slave_ip="192.168.100.2", tcp_port=5005)

        logger.info("FramePutter đã được khởi tạo thành công")

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        """
        Đọc đồng thời frame từ camera RGB và ToF.
        Xử lý frame RGB, lưu tất cả dữ liệu và đưa đường dẫn vào hàng đợi.
        """
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(self.rgb_camera_id)
        self.prev_frame = None

        # Tạo các thư mục lưu trữ nếu chưa có
        for d in [self.rgb_frame_dir, self.depth_data_dir, self.amp_frame_dir]:
            os.makedirs(d, exist_ok=True)

        if not cap.isOpened():
            logger.error(f"❌ Không mở được camera RGB với ID: {self.rgb_camera_id}")
            await frame_queue.put(None) # Báo cho consumer dừng lại
            return

        self.start_time = time.perf_counter()
        logger.info(f"Đã mở camera RGB (ID: {self.rgb_camera_id}) và sẵn sàng lấy dữ liệu ToF.")
        
        try:
            while not self.stop_event.is_set():
                if frame_queue.full():
                    logger.warning("Queue đã đầy, tạm dừng đọc frame.")
                    await asyncio.sleep(0.1)
                    continue

                # --- ĐỌC DỮ LIỆU TỪ 2 CAMERA ĐỒNG THỜI ---
                # Chạy 2 hàm blocking trong các luồng riêng biệt và chờ cả hai hoàn thành
                try:
                    (ret, frame), (depth_data, amp_frame) = await asyncio.gather(
                        loop.run_in_executor(None, cap.read),
                        loop.run_in_executor(None, self.slave.request_and_receive_tof_frames)
                    )
                except Exception as e:
                    logger.error(f"Lỗi khi chạy song song tác vụ camera: {e}")
                    await asyncio.sleep(1)
                    continue
                # --------------------------------------------

                # Kiểm tra kết quả từ cả hai camera
                if not ret or frame is None:
                    logger.warning("Không nhận được frame từ camera RGB.")
                    await asyncio.sleep(0.01)
                    continue
                
                if depth_data is None or amp_frame is None:
                    logger.warning("Không nhận được frame từ camera ToF (Slave).")
                    await asyncio.sleep(0.5) # Chờ một chút trước khi thử lại
                    continue
                
                # Xử lý frame RGB (ví dụ: loại bỏ nền)
                _, foreground, _ = await asyncio.to_thread(
                    self.background_remover.remove_background, frame
                )

                # Kiểm tra thay đổi so với frame trước
                if self.prev_frame is not None and \
                   cv2.absdiff(foreground, self.prev_frame).sum() < 10000:
                    await asyncio.sleep(0.1)
                    continue
                
                self.prev_frame = foreground.copy()
                
                # Tạo tên file nhất quán cho bộ dữ liệu
                base_filename = f"capture_{self.frame_count:06d}"

                # --- LƯU TẤT CẢ CÁC FRAME VÀO ĐĨA ---
                # 1. Lưu frame RGB
                rgb_path = os.path.join(self.rgb_frame_dir, f"{base_filename}.jpg")
                await loop.run_in_executor(None, cv2.imwrite, rgb_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                # 2. Lưu frame Amplitude (dưới dạng .npy hoặc .jpg nếu muốn)
                amp_path = os.path.join(self.amp_frame_dir, f"{base_filename}.jpg")
                await loop.run_in_executor(None, cv2.imwrite, amp_path, amp_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                self.frame_count += 1
                
                # --- ĐƯA DỮ LIỆU VÀO QUEUE ---
                # Gói tất cả các đường dẫn vào một tuple
                # Định dạng: (cam_index, rgb_path, depth_path, amp_path)
                data_packet = (self.cam_index, rgb_path, amp_path, depth_data)
                
                try:
                    await asyncio.wait_for(frame_queue.put(data_packet), timeout=2.0)
                    logger.debug(f"Đã enqueue gói dữ liệu #{self.frame_count - 1}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout khi enqueue gói dữ liệu #{self.frame_count - 1}")

                self._update_fps()

        finally:
            logger.info("Dừng vòng lặp, đang dọn dẹp tài nguyên...")
            cap.release()
            self.slave.close() # Quan trọng: đóng kết nối socket
            await frame_queue.put(None) # Gửi sentinel để consumer biết dừng
            logger.info("Đã dừng camera, đóng kết nối slave và enqueue sentinel.")

    def _update_fps(self):
        # Hàm này không thay đổi
        if self.start_time:
            elapsed_time = time.perf_counter() - self.start_time
            if elapsed_time > 1: # Cập nhật mỗi giây
                self.fps = self.frame_count / elapsed_time
                logger.info(f"Current FPS: {self.fps:.2f}")
                # Reset để tính toán cho giây tiếp theo
                self.frame_count = 0
                self.start_time = time.perf_counter()

    async def stop(self):
        self.stop_event.set()

async def start_putter(frame_queue: asyncio.Queue):
    """Start putter to push frames into queue."""
    logger.info("Starting putter...")
    frame_putter = FramePutter()
    try:
        await frame_putter.put_frames_queue(frame_queue)
    except asyncio.CancelledError:
        logger.info("Putter task was cancelled.")
        frame_putter.stop()
    except Exception as e:
        logger.error(f"Error in putter task: {e}")
        frame_putter.stop()
    finally:
        frame_putter.stop()