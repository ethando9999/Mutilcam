#!/usr/bin/env python3
import asyncio
import threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging
import os

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo GStreamer
Gst.init(None)

class GStreamerFramePutter:
    def __init__(self, device='/dev/video0', width=2592, height=1944, framerate=30, queue_maxsize=100):
        """Khởi tạo GStreamerFramePutter với pipeline và queue."""
        self.pipeline = Gst.parse_launch(
            f"v4l2src device={device} ! image/jpeg, width={width}, height={height}, framerate={framerate}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        self.appsink = self.pipeline.get_by_name("sink")

        # Queue cho frame đã lọc
        self.frame_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.loop = asyncio.get_event_loop()

        # Background remover
        from utils.rmbg_mog import BackgroundRemover
        self.background_remover = BackgroundRemover()
        self.prev_foreground = None

        # Thread pool cho xử lý nền
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Kết nối callback
        self.appsink.connect("new-sample", self.on_new_sample)

    def on_new_sample(self, sink):
        """Callback khi có sample mới từ GStreamer."""
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))

        # Gửi vào thread pool để xử lý background & so sánh
        self.executor.submit(self.process_frame, frame)
        return Gst.FlowReturn.OK

    def process_frame(self, frame):
        """Xử lý background, lọc theo thay đổi, rồi enqueue frame gốc."""
        _, fg, _ = self.background_remover.remove_background(frame)

        if self.prev_foreground is not None:
            diff = cv2.absdiff(fg, self.prev_foreground).sum()
            if diff < 10000:
                logger.debug("Không có thay đổi đáng kể, bỏ qua frame")
                return

        self.prev_foreground = fg.copy()
        asyncio.run_coroutine_threadsafe(self.enqueue_frame(frame), self.loop)

    async def enqueue_frame(self, frame):
        """Đẩy frame vào queue với timeout."""
        try:
            await asyncio.wait_for(self.frame_queue.put(frame), timeout=2.0)
            logger.debug("Đã đẩy frame vào queue")
        except asyncio.TimeoutError:
            logger.warning("Queue đầy, bỏ frame")

    def start(self):
        """Khởi động GStreamer và GLib MainLoop."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self.glib_loop = GLib.MainLoop()
        self.glib_thread = threading.Thread(target=self.glib_loop.run, daemon=True)
        self.glib_thread.start()
        logger.info("GStreamer capture started")

    def stop(self):
        """Dừng GStreamer và thread pool."""
        self.pipeline.set_state(Gst.State.NULL)
        self.glib_loop.quit()
        self.glib_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("GStreamer capture stopped")

    def start_recording(self, output_path: str, fourcc_str='avc1', fps=30):
        """
        Bắt đầu thread ghi video MP4.
        Writer được khởi tạo khi có frame đầu tiên.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._recording = True

        def _record_loop():
            writer = None
            logger.info(f"Bắt đầu ghi video vào {output_path}")
            while self._recording:
                future = asyncio.run_coroutine_threadsafe(self.frame_queue.get(), self.loop)
                try:
                    frame = future.result(timeout=0.1)
                except FutureTimeoutError:
                    continue

                if frame is None:
                    break

                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        logger.error(f"Không thể tạo VideoWriter với {output_path}")
                        return

                writer.write(frame)

            if writer:
                writer.release()
            logger.info("Đã dừng ghi video")

        self._record_thread = threading.Thread(target=_record_loop, daemon=True)
        self._record_thread.start()

    def stop_recording(self):
        """Dừng ghi video, gửi sentinel và chờ thread kết thúc."""
        self._recording = False
        asyncio.run_coroutine_threadsafe(self.frame_queue.put(None), self.loop)
        if hasattr(self, '_record_thread'):
            self._record_thread.join()

async def start_putter(external_queue: asyncio.Queue):
    """
    Chạy GStreamerFramePutter và forward các frame lọc sẵn từ frame_queue
    sang external_queue do user cung cấp.
    """
    putter = GStreamerFramePutter()
    putter.start()
    try:
        while True:
            frame = await putter.frame_queue.get()
            if frame is None:
                break
            try:
                await asyncio.wait_for(external_queue.put(frame), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("External queue đầy, bỏ frame")
    finally:
        putter.stop()
        await external_queue.put(None)
        logger.info("Đã dừng GStreamer capture và forward")

if __name__ == "__main__":
    async def main():
        external_queue = asyncio.Queue(maxsize=100)
        putter = GStreamerFramePutter()
        putter.start()

        # Ghi video MP4
        putter.start_recording("videos/output.mp4", fourcc_str='avc1', fps=30)

        # Chạy 60s rồi dừng
        await asyncio.sleep(60)

        putter.stop_recording()
        putter.stop()

    asyncio.run(main())
