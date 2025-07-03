import asyncio
import cv2
import os
from utils.rmbg_mog import BackgroundRemover
from utils.config_camera import CameraHandler

from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)

class FramePutter:
    def __init__(self):
        self.stop_event = asyncio.Event()
        self.fps = 0
        self.prev_frame = None
        self.frame_count = 0
        self.start_time = None
        self.background_remover = BackgroundRemover()
        self.cam0 = CameraHandler(camera_index=0)
        logger.info("Start FramePutter successfully")

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        """Xử lý khung hình và đẩy vào hàng đợi một cách bất đồng bộ."""
        self.start_time = asyncio.get_event_loop().time()
        
        while not self.stop_event.is_set():
            try:
                # Kiểm tra queue đầy 
                if frame_queue.full():
                    # logger.warning("Frame queue is full. Pausing frame processing...")
                    await asyncio.sleep(0.2)
                    continue

                # Lấy original_frame từ capture_lores_frame
                lores_frame =  self.cam0.capture_lores_frame()
                if lores_frame is None:
                    logger.error("Không thể chụp ảnh lores từ camera.")
                    await asyncio.sleep(0.1)
                    continue

                # logger.info("Bắt đầu xử lý background removal...")
                # Xử lý background
                _, foreground, mask_boxes = self.background_remover.remove_background(lores_frame)
                # logger.debug(f"Đã tìm thấy {len(mask_boxes)} bounding boxes")

                # Kiểm tra thay đổi so với prev_frame
                if self.prev_frame is not None and cv2.absdiff(foreground, self.prev_frame).sum() < 10000:
                    # logger.warning("Không có thay đổi đáng kể, bỏ qua frame")
                    await asyncio.sleep(0.1) 
                    continue

                # Cập nhật prev_frame
                self.prev_frame = foreground.copy()
                
                # logger.debug("Bắt đầu chụp ảnh main...")
                main_frame =  self.cam0.capture_main_frame()
                if main_frame is None:
                    logger.error("Không thể chụp ảnh main từ camera.")
                    await asyncio.sleep(0.1)
                    continue

                original_height, original_width = main_frame.shape[:2]
                original_size = main_frame.nbytes
                # logger.info(f"original_frame size (width x height): ({original_width}x{original_height})")

                # Sử dụng try_put để tránh blocking 
                try:
                    await asyncio.wait_for(frame_queue.put(main_frame), timeout=0.5)
                    # logger.info(f"Đã đưa frame {self.frame_count} vào queue thành công (width x height): ({original_width}x{original_height})")
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

    async def put_frame_from_video(self, frame_queue: asyncio.Queue, video_path="/home/ubuntu/orangepi/python/data/output_4k_video.mp4"):
        """Read frames from a video file and put them into the frame queue."""
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        self.start_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.info(f"End of video or invalid frame at index {frame_index}")
                    break

                # Tăng timeout lên 5 giây để giảm nguy cơ mất khung hình
                try:
                    await asyncio.wait_for(frame_queue.put(frame), timeout=15.0)
                    logger.info(f"Putting frame {frame_index} from video")
                    frame_index += 1
                    self._update_fps()
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout putting frame {frame_index} into queue")
                    continue

        except Exception as e:
            logger.error(f"Error reading video: {e}")
        finally:
            cap.release()
            await frame_queue.put(None)
            logger.info("Video capture released")

    def _update_fps(self):
        """Update FPS calculation.""" 
        self.frame_count += 1
        current_time = asyncio.get_event_loop().time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            logger.info(f"FPS putting: {self.fps:.2f}")
            self.start_time = current_time
            self.frame_count = 0

    def stop(self):
        """Stop frame processing."""
        self.stop_event.set()

async def start_putter(frame_queue: asyncio.Queue):
    """Start putter to push frames into queue."""
    logger.info("Starting putter...")
    frame_putter = FramePutter()
    try:
        video_name = "video.mp4"
        video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        video_path = os.path.join(video_path, video_name)
        await frame_putter.put_frame_from_video(frame_queue, video_path)
        # await frame_putter.put_frames_queue(frame_queue)
    except asyncio.CancelledError:
        logger.info("Putter task was cancelled.")
        frame_putter.stop()
    except Exception as e:
        logger.error(f"Error in putter task: {e}")
        frame_putter.stop()
    finally:
        frame_putter.stop()