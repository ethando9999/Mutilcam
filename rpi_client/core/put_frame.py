from utils.camera_log import get_logger
from utils.config_camera import CameraHandler
from utils.rmbg_mog import BackgroundRemover
import asyncio
import cv2

logger = get_logger(__name__)

class FramePutter:
    def __init__(self):
        self.cam0 = CameraHandler(camera_index=0)
        self.cam1 = CameraHandler(camera_index=1)
        self.background_remover = BackgroundRemover()
        self.stop_event = asyncio.Event()
        self.fps = 0
        self.prev_frame = None
        self.frame_count = 0
        self.start_time = None

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        """Xử lý khung hình và đẩy vào hàng đợi một cách bất đồng bộ."""
        self.start_time = asyncio.get_event_loop().time()
        
        while not self.stop_event.is_set():
            try:
                # Kiểm tra queue đầy 
                if frame_queue.full():
                    logger.warning("Frame queue is full. Pausing frame processing...")
                    await asyncio.sleep(0.2)
                    continue

                # Lấy original_frame từ capture_lores_frame
                lores_frame =  self.cam1.capture_lores_frame()
                if lores_frame is None:
                    logger.error("Không thể chụp ảnh lores từ camera.")
                    await asyncio.sleep(0.1)
                    continue

                logger.debug("Bắt đầu xử lý background removal...")
                # Xử lý background
                _, foreground, mask_boxes = self.background_remover.remove_background(lores_frame)
                logger.debug(f"Đã tìm thấy {len(mask_boxes)} bounding boxes")

                # Kiểm tra thay đổi so với prev_frame
                if self.prev_frame is not None and cv2.absdiff(foreground, self.prev_frame).sum() < 10000:
                    logger.debug("Không có thay đổi đáng kể, bỏ qua frame")
                    await asyncio.sleep(0.1)
                    continue

                # Cập nhật prev_frame
                self.prev_frame = foreground.copy()
                
                logger.debug("Bắt đầu chụp ảnh main...")
                main_frame =  self.cam0.capture_main_frame()
                if main_frame is None:
                    logger.error("Không thể chụp ảnh main từ camera.")
                    await asyncio.sleep(0.1)
                    continue

                original_height, original_width = main_frame.shape[:2]
                original_size = main_frame.nbytes
                logger.info(f"original_frame size (width x height): ({original_width}x{original_height})")

                # Sử dụng try_put để tránh blocking 
                try:
                    await asyncio.wait_for(frame_queue.put(main_frame), timeout=0.5)
                    logger.debug(f"Đã đưa frame {self.frame_count} vào queue thành công")
                except asyncio.TimeoutError:
                    logger.warning("Timeout khi đưa frame vào queue")
                    continue

                self._update_fps()

            except Exception as e:
                logger.error(f"Lỗi trong put_frames_queue: {e}")
                if self.prev_frame is not None:
                    del self.prev_frame
                await asyncio.sleep(0.1)
                continue

    def _update_fps(self):
        """Cập nhật FPS."""
        self.frame_count += 1
        current_time = asyncio.get_event_loop().time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            logger.info(f"FPS putting: {self.fps:.2f}")
            self.start_time = current_time
            self.frame_count = 0

    def stop(self):
        """Dừng xử lý frame."""
        self.stop_event.set()

async def start_putter(frame_queue: asyncio.Queue):
    """
    Khởi động putter để đẩy frame vào queue.
    
    Args:
        frame_queue: Queue chứa các frame
    """
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