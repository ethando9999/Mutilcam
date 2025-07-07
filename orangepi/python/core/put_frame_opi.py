from utils.logging_python_orangepi import get_logger
import asyncio
import cv2
import time 
import os

from utils.rmbg_mog import BackgroundRemover
# from .call_slaver import CallSlave

logger = get_logger(__name__)

class FramePutter:
    def __init__(self):
        self.stop_event = asyncio.Event()
        self.fps = 0
        self.prev_frame = None
        self.frame_count = 0
        self.start_time = None
        self.frame_dir = "frames"
        self.cam_index = 1 
        self.background_remover = BackgroundRemover()
        # self.slave = CallSlave()
        logger.info("Start FramePutter successfully") 

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        """
        Đọc frame từ camera, loại bỏ nền và chỉ lưu những frame có thay đổi đáng kể.
        Lưu khung hình vào thẻ nhớ SD nội bộ, rồi đưa đường dẫn vào hàng đợi.
        Kết thúc: release camera và đặt None làm sentinel cho consumer.
        """
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(0)
        self.prev_frame = None

        # Thư mục lưu frame
        os.makedirs(self.frame_dir, exist_ok=True)

        if not cap.isOpened():
            raise RuntimeError("❌ Không mở được camera")

        self.start_time = time.perf_counter()
        logger.info("Đã mở được camera")
        try:
            while not self.stop_event.is_set():
                # Nếu queue đầy thì chờ
                if frame_queue.full():
                    logger.warning("Queue đã đầy, tạm dừng đọc frame")
                    await asyncio.sleep(0.1)
                    continue

                # Đọc frame không block event loop
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret or frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Loại bỏ nền
                _, foreground, _ = await asyncio.to_thread(
                    self.background_remover.remove_background, frame
                )

                # Kiểm tra thay đổi
                if self.prev_frame is not None and \
                cv2.absdiff(foreground, self.prev_frame).sum() < 10000:
                    await asyncio.sleep(0.1)
                    continue

                self.prev_frame = foreground.copy()

                # Lưu khung hình gốc vào thẻ nhớ SD
                frame_path = os.path.join(
                    self.frame_dir,
                    f"frame_{self.frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # logger.info(f"Đã lưu frame #{self.frame_count} => {frame_path}")
                self.frame_count += 1

                # Đưa đường dẫn vào queue với timeout
                try:
                    await asyncio.wait_for(frame_queue.put((self.cam_index, frame_path)), timeout=2.0)
                    # await time.sleep(0.001)
                    # logger.info(f"Enqueued đường dẫn frame #{self.frame_count - 1}")
                except asyncio.TimeoutError:
                    logger.warning("Không enqueue được, queue đầy")

                # Cập nhật FPS (nếu có)
                self._update_fps()

        finally:
            cap.release()
            # gửi sentinel để consumer biết dừng
            await frame_queue.put(None)
            logger.info("Đã dừng camera và enqueue sentinel None")

    async def put_frame_from_video(self, frame_queue: asyncio.Queue, video_path="/home/ubuntu/orangepi/python/data/output_4k_video.mp4"):
        """
        Đọc frame từ video, lưu vào file tạm và đưa đường dẫn vào hàng đợi.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Không thể mở file video tại: {video_path}")
            # Gửi tín hiệu kết thúc ngay nếu không mở được video
            await frame_queue.put(None)
            return

        frame_index = 0
        self.start_time = asyncio.get_event_loop().time()
        
        # --- Tạo thư mục lưu frame nếu chưa có ---
        os.makedirs(self.frame_dir, exist_ok=True)
        logger.info(f"Các khung hình sẽ được lưu tạm tại: {os.path.abspath(self.frame_dir)}")

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.info(f"Hết video hoặc khung hình không hợp lệ tại index {frame_index}")
                    break

                # --- Logic chính đã được sửa ---
                # 1. Tạo đường dẫn file cho khung hình hiện tại
                frame_path = os.path.join(
                    self.frame_dir,
                    f"frame_{frame_index:06d}.jpg"
                )

                # 2. Lưu khung hình xuống file (bất đồng bộ để không block)
                # Dùng `await asyncio.to_thread` là cách hiện đại và an toàn hơn
                save_success = await asyncio.to_thread(cv2.imwrite, frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                if not save_success:
                    logger.error(f"Không thể lưu khung hình xuống file: {frame_path}")
                    frame_index += 1
                    continue

                # 3. Đưa vào queue đúng định dạng (camera_id, frame_path)
                item_to_put = (self.cam_index, frame_path)
                
                try:
                    # Đặt timeout hợp lý để tránh bị block quá lâu
                    await asyncio.wait_for(frame_queue.put(item_to_put), timeout=5.0)
                    logger.info(f"Đã đưa vào hàng đợi: {item_to_put}")
                    frame_index += 1
                    self._update_fps()
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout khi đưa khung hình {frame_index} vào hàng đợi. Hàng đợi có thể đã đầy.")
                    # Xóa file đã lưu nếu không đưa vào queue được để tránh rác
                    os.remove(frame_path)
                    # Chờ một chút trước khi thử lại
                    await asyncio.sleep(0.1)
                    continue

        except Exception as e:
            logger.error(f"Lỗi khi đọc video: {e}", exc_info=True)
        finally:
            cap.release()
            # Gửi tín hiệu kết thúc cho consumer
            await frame_queue.put(None)
            logger.info("Đã giải phóng video và gửi tín hiệu kết thúc.")

    async def put_frames_queue_from_camera(self, frame_queue: asyncio.Queue):
        """Đọc frame từ camera và put vào queue cho tới khi stop_event được set.
           Kết thúc: release camera và đặt None làm sentinel cho consumer."""
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(0)
        self.prev_frame = None
        
        if not cap.isOpened():
            raise RuntimeError("❌ Không mở được camera")

        self.start_time = time.perf_counter()
        logger.info(f"Đã mở được camera")
        try:
            while not self.stop_event.is_set(): 
                # Đọc frame trong thread pool (không block event-loop)
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    await asyncio.sleep(0.01)           # thử lại nhẹ nhàng
                    continue

                _, foreground, mask_boxes = self.background_remover.remove_background(frame)
                if self.prev_frame is not None and cv2.absdiff(foreground, self.prev_frame).sum() < 10000:
                    # logger.warning("Không có thay đổi đáng kể, bỏ qua frame")
                    await asyncio.sleep(0.1)
                    continue

                self.prev_frame = foreground.copy()

                await frame_queue.put(frame)            # chuyển frame cho consumer
                logger.info(f"Putting frame {self.frame_count} from video")

                # Cập nhật FPS mỗi 30 frame
                self._update_fps()
        finally:
            cap.release()
            await frame_queue.put(None) 

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
        video_name = "output_4k_video.mp4"
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