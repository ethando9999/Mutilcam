import os
import cv2
import asyncio
import shutil
from utils.logging_python_orangepi import get_logger
# from utils.config_camera import CameraHandler
from utils.rmbg_mog import BackgroundRemover

logger = get_logger(__name__)

frame_dir = "frames"

class FramePutter:
    def __init__(self, camera_indices=[0], frame_dir=frame_dir):
        self.stop_event = asyncio.Event()
        self.fps = 0.0
        self.fps_avg = 0.0
        self.frame_count = 0
        self.call_count = 0
        self.prev_frame = {}
        self.start_time = None
        self.background_remover = BackgroundRemover()
        self.cameras = {}
        self.frame_dir = frame_dir
        
        # Kiểm tra và tạo thư mục nếu chưa tồn tại
        if not os.path.exists(frame_dir):
            try:
                os.makedirs(frame_dir, exist_ok=True)
            except PermissionError:
                logger.error(f"Không thể tạo thư mục {frame_dir}: Permission denied")
                raise
        elif not os.access(frame_dir, os.W_OK):
            logger.error(f"Thư mục {frame_dir} không thể ghi")
            raise PermissionError(f"Thư mục {frame_dir} không thể ghi")
        else:
            shutil.rmtree(frame_dir)
            os.makedirs(frame_dir, exist_ok=True)
        
        # for idx in camera_indices:
        #     self.cameras[idx] = CameraHandler(camera_index=idx)
        #     logger.info(f"Khởi tạo camera {idx} cho FramePutter")

    def _update_fps(self):
        self.frame_count += 1
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + self.fps) / (self.call_count + 1)
        self.call_count += 1

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        self.start_time = asyncio.get_event_loop().time()
        
        while not self.stop_event.is_set():
            for cam_idx, cam in self.cameras.items():
                try:
                    if frame_queue.full():
                        logger.warning("Hàng đợi khung hình đã đầy. Tạm dừng xử lý...")
                        await asyncio.sleep(0.1)
                        continue

                    lores_frame = cam.capture_lores_frame()
                    if lores_frame is None or lores_frame.size == 0:
                        logger.error(f"Khung hình lores không hợp lệ từ camera {cam_idx}.")
                        await asyncio.sleep(0.1)
                        continue

                    _, foreground, mask_boxes = await asyncio.to_thread(
                        self.background_remover.remove_background, lores_frame
                    )

                    if cam_idx in self.prev_frame and cv2.absdiff(foreground, self.prev_frame[cam_idx]).sum() < 10000:
                        logger.debug(f"Không phát hiện thay đổi đáng kể từ camera {cam_idx}, bỏ qua.")
                        await asyncio.sleep(0.1)
                        continue

                    self.prev_frame[cam_idx] = foreground.copy() if foreground is not None else None
                    
                    main_frame = cam.capture_main_frame()
                    if main_frame is None or main_frame.size == 0:
                        logger.error(f"Khung hình chính không hợp lệ từ camera {cam_idx}.")
                        await asyncio.sleep(0.1)
                        continue

                    # Lưu khung hình vào thẻ nhớ SD nội bộ
                    frame_path = os.path.join(self.frame_dir, f"frame_{cam_idx}_{self.frame_count}.jpg")
                    cv2.imwrite(frame_path, main_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])  # Nén JPEG với chất lượng 85%

                    # Đưa đường dẫn vào hàng đợi
                    await asyncio.wait_for(frame_queue.put((cam_idx, frame_path)), timeout=2.0)
                    logger.debug(f"Đưa đường dẫn khung hình {self.frame_count} từ camera {cam_idx} vào hàng đợi.")
                    # await asyncio.sleep(0.001)
                    self._update_fps()

                except Exception as e:
                    logger.error(f"Lỗi trong put_frames_queue cho camera {cam_idx}: {e}", exc_info=True)
                    await asyncio.sleep(0.1)
                    continue

    async def put_frame_from_video(
        self, 
        frame_queue: asyncio.Queue, 
        video_path="/home/ubuntu/orangepi/python/data/output_4k_video.mp4"
    ):
        """Read frames from a video file, save to SD card and put file path into queue."""
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        self.start_time = asyncio.get_event_loop().time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.info(f"End of video or invalid frame at index {frame_index}")
                    break

                # Lưu khung hình vào thẻ SD
                frame_path = os.path.join(self.frame_dir, f"frame_video_{frame_index}.jpg")
                cv2.imwrite(
                    frame_path,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]  # Chất lượng JPEG
                )

                # Đưa đường dẫn file vào hàng đợi
                try:
                    await asyncio.wait_for(
                        frame_queue.put((frame_index, frame_path)), 
                        timeout=15.0
                    )
                    logger.debug(f"Put video frame {frame_index} to queue: {frame_path}")
                    self._update_fps()
                    frame_index += 1
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout putting frame {frame_index} into queue")
                    continue

                # Nghỉ nhẹ để không chặn event loop
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error reading video: {e}", exc_info=True)
        finally:
            cap.release()
            await frame_queue.put(None)  # tín hiệu kết thúc
            logger.info("Video capture released")

    def stop(self):
        self.stop_event.set()
        for cam_idx, cam in self.cameras.items():
            cam.stop_camera()
            logger.info(f"Tài nguyên camera {cam_idx} đã được giải phóng")

async def start_putter(frame_queue: asyncio.Queue, camera_indices=[0]):
    putter = FramePutter(camera_indices=camera_indices)
    try:
        # await putter.put_frames_queue(frame_queue) 
        video_name = "output_4k_video.mp4"
        video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        video_path = os.path.join(video_path, video_name)
        await putter.put_frame_from_video(frame_queue, video_path)
    except asyncio.CancelledError:
        putter.stop()
        logger.info("Đã dừng FramePutter một cách an toàn.")