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
import os
from concurrent.futures import TimeoutError as FutureTimeoutError

logger = get_logger(__name__)

# Khởi tạo GStreamer
Gst.init(None)

class GStreamerFramePutter:
    def __init__(self, device='/dev/video0', width=2592, height=1944, framerate=30, queue_maxsize=100):
        """Khởi tạo GStreamerFramePutter với pipeline và queue."""
        # Tạo pipeline GStreamer
        self.pipeline = Gst.parse_launch(
            f"v4l2src device={device} ! image/jpeg, width={width}, height={height}, framerate={framerate}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        self.appsink = self.pipeline.get_by_name("sink")
        
        # Khởi tạo queue và các thành phần khác
        self.frame_queue = asyncio.Queue(maxsize=queue_maxsize)  # Queue giới hạn kích thước
        self.loop = asyncio.get_event_loop()
        self.background_remover = BackgroundRemover()  # Khởi tạo BackgroundRemover
        self.prev_foreground = None  # Lưu foreground trước đó để so sánh
        self.appsink.connect("new-sample", self.on_new_sample)  # Kết nối callback
        self.executor = ThreadPoolExecutor(max_workers=4)  # Thread pool cho xử lý bất đồng bộ

    def on_new_sample(self, sink):
        """Xử lý mẫu mới từ appsink."""
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")
        data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # Gửi frame vào thread pool để xử lý bất đồng bộ
        self.executor.submit(self.process_frame, frame)
        return Gst.FlowReturn.OK

    def process_frame(self, frame):
        """Xử lý frame: loại bỏ nền và kiểm tra thay đổi."""
        # Loại bỏ nền
        _, foreground, _ = self.background_remover.remove_background(frame)
        
        # Kiểm tra sự thay đổi so với foreground trước đó
        if self.prev_foreground is not None:
            diff = cv2.absdiff(foreground, self.prev_foreground).sum()
            if diff < 10000:  # Ngưỡng thay đổi (có thể điều chỉnh)
                logger.debug("Không có thay đổi đáng kể, bỏ qua frame")
                return
        
        # Cập nhật foreground trước đó
        self.prev_foreground = foreground.copy()
        
        # Đẩy frame gốc vào queue nếu có thay đổi đáng kể
        asyncio.run_coroutine_threadsafe(self.enqueue_frame(frame), self.loop)

    async def enqueue_frame(self, frame):
        """Đẩy frame vào queue với timeout."""
        try:
            await asyncio.wait_for(self.frame_queue.put(frame), timeout=2.0)
            logger.debug("Đã đẩy khung hình gốc có thay đổi đáng kể vào queue")
        except asyncio.TimeoutError:
            logger.warning("Queue đầy, bỏ qua khung hình từ camera")

    def start(self):
        """Khởi động pipeline và GLib loop."""
        self.pipeline.set_state(Gst.State.PLAYING)
        self.glib_loop = GLib.MainLoop()
        self.glib_thread = threading.Thread(target=self.glib_loop.run)
        self.glib_thread.start()

    def stop(self):
        """Dừng pipeline và GLib loop."""
        self.pipeline.set_state(Gst.State.NULL)
        self.glib_loop.quit()
        self.glib_thread.join()
        self.executor.shutdown(wait=True)

    def start_recording(self, output_path: str, fourcc_str='avc1', fps=30):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._recording = True

        def _record_loop():
            writer = None
            logger.info(f"Bắt đầu ghi video vào {output_path}")

            while self._recording:
                # Khởi một future để lấy frame từ asyncio.Queue
                future = asyncio.run_coroutine_threadsafe(self.frame_queue.get(), self.loop)
                try:
                    # Chờ tối đa 0.1s
                    frame = future.result(timeout=0.1)
                except FutureTimeoutError:
                    # Nếu timeout, quay lại vòng while
                    continue

                # Nếu sentinel None, thoát loop
                if frame is None:
                    break

                # Khởi tạo writer từ frame đầu tiên
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
        self._recording = False
        # Gửi sentinel để thread nhận ra phải dừng
        asyncio.run_coroutine_threadsafe(self.frame_queue.put(None), self.loop)
        if hasattr(self, '_record_thread'):
            self._record_thread.join()

async def start_putter(frame_queue: asyncio.Queue):
    """Chạy GStreamerFramePutter và chuyển frame từ queue nội bộ sang queue ngoài."""
    putter = GStreamerFramePutter()
    putter.start()
    try:
        while True:
            frame = await putter.frame_queue.get()
            if frame is None:
                break
            try:
                await asyncio.wait_for(frame_queue.put(frame), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Queue đầy, bỏ frame từ camera")
    finally:
        putter.stop()
        await frame_queue.put(None)  # Đặt sentinel để báo hiệu dừng
        logger.info("Đã dừng GStreamer capture")

if __name__ == "__main__":
    async def main():
        external_queue = asyncio.Queue(maxsize=100)
        putter = GStreamerFramePutter()
        putter.start()
        # Bắt đầu ghi video, lưu vào videos/output.avi
        putter.start_recording("videos/output.mp4", fourcc_str='avc1', fps=30)

        # Đồng thời bạn có thể xử lý/external_queue như bình thường
        # Ví dụ dừng sau 60 giây:
        await asyncio.sleep(60)

        # Dừng capture và ghi
        putter.stop_recording()
        putter.stop()

    asyncio.run(main())
