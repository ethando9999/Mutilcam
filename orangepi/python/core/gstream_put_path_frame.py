import os
import asyncio
import threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from utils.rmbg_mog import BackgroundRemover
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Khởi tạo GStreamer
Gst.init(None)

class GStreamerFramePutter:
    def __init__(
        self,
        device: str = '/dev/video0',
        width: int = 2592,
        height: int = 1944,
        framerate: int = 30,
        queue_maxsize: int = 100,
        frame_dir: str = 'frames'
    ):
        """Khởi tạo pipeline, queue và thư mục lưu frames."""
        # Pipeline GStreamer
        self.pipeline = Gst.parse_launch(
            f"v4l2src device={device} ! image/jpeg, width={width}, height={height}, "
            f"framerate={framerate}/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self.on_new_sample)

        # Queue nội bộ chứa đường dẫn file ảnh
        self.frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_maxsize)
        self.loop = asyncio.get_event_loop()

        # Background remover và so sánh
        self.background_remover = BackgroundRemover()
        self.prev_foreground: np.ndarray | None = None

        # Thread pool xử lý bất đồng bộ
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Thư mục lưu frame và biến đếm
        self.frame_dir = frame_dir
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0
        self.cam_index = 1

    def on_new_sample(self, sink) -> Gst.FlowReturn:
        """Callback khi có sample mới, đẩy vào thread pool."""
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        self.executor.submit(self.process_frame, frame)
        return Gst.FlowReturn.OK

    def process_frame(self, frame: np.ndarray):
        """Loại bỏ nền, kiểm tra thay đổi, lưu file và enqueue đường dẫn."""
        _, foreground, _ = self.background_remover.remove_background(frame)

        # Nếu không có frame trước để so sánh, chỉ cập nhật
        if self.prev_foreground is not None:
            diff = cv2.absdiff(foreground, self.prev_foreground).sum()
            if diff < 10000:
                logger.info("Không có thay đổi đáng kể, bỏ qua frame")
                return

        # Cập nhật foreground trước đó
        self.prev_foreground = foreground.copy()

        # Lưu khung hình gốc vào thẻ nhớ
        frame_path = os.path.join(self.frame_dir, f"frame_{self.frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.frame_count += 1
        logger.info(f"Đã lưu frame vào {frame_path}")

        # Đưa đường dẫn vào queue
        asyncio.run_coroutine_threadsafe(
            self.enqueue_path(frame_path),
            self.loop
        )

    async def enqueue_path(self, path: str):
        """Đưa đường dẫn file vào queue với timeout."""
        try:
            await asyncio.wait_for(self.frame_queue.put(self.cam_index, path), timeout=2.0)
            logger.info(f"Đã enqueue đường dẫn: {path}")
        except asyncio.TimeoutError:
            logger.warning("Queue đầy, bỏ qua đường dẫn frame")

    def start(self):
        """Chạy pipeline và GLib MainLoop trên thread riêng."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self.glib_loop = GLib.MainLoop()
        self.glib_thread = threading.Thread(target=self.glib_loop.run, daemon=True)
        self.glib_thread.start()
        logger.info("GStreamer pipeline đã khởi động")

    def stop(self):
        """Dừng pipeline, GLib loop và executor."""
        self.pipeline.set_state(Gst.State.NULL)
        self.glib_loop.quit()
        self.glib_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("GStreamer pipeline đã dừng")


async def start_putter(frame_queue: asyncio.Queue[str]):
    """
    Khởi GStreamerFramePutter, nhận các đường dẫn file từ putter.frame_queue
    và đẩy tiếp vào frame_queue bên ngoài.
    """
    putter = GStreamerFramePutter()
    putter.start()
    try:
        while True:
            path = await putter.frame_queue.get()
            if path is None:
                break
            try:
                await asyncio.wait_for(frame_queue.put(path), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Queue ngoài đầy, bỏ qua đường dẫn frame")
    finally:
        putter.stop()
        await frame_queue.put(None)
        logger.info("Đã dừng GStreamer capture")
