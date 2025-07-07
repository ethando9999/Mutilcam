# modules/frame_producers.py
import asyncio
import threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import socket
import json
import struct

from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)
Gst.init(None)

# --- Phần Producer cho Camera RGB (Dựa trên GStreamer) ---
class GstRGBProducer:
    def __init__(self, frame_queue: asyncio.Queue, device='/dev/rgb_cam', width=640, height=480, framerate=30):
        self.loop = asyncio.get_event_loop()
        self.frame_queue = frame_queue
        self.pipeline = Gst.parse_launch(
            f"v4l2src device={device} io-mode=4 ! image/jpeg, width={width}, height={height}, framerate={framerate}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink emit-signals=true max-buffers=5 drop=true"
        )
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_new_sample)

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        frame = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.uint8).reshape((h, w, 3))
        
        try:
            # Dùng non-blocking put để không làm treo luồng GStreamer
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("RGB frame queue is full, dropping a frame.")
        return Gst.FlowReturn.OK

    async def run(self):
        logger.info("Starting GStreamer RGB Producer...")
        self.pipeline.set_state(Gst.State.PLAYING) 
        # Chạy GLib loop trong một thread riêng để không block asyncio
        glib_loop = GLib.MainLoop()
        glib_thread = threading.Thread(target=glib_loop.run)
        glib_thread.start()
        
        try:
            await asyncio.Event().wait()  # Chạy vô hạn cho đến khi bị hủy
        finally:
            logger.info("Stopping GStreamer RGB Producer...")
            self.pipeline.set_state(Gst.State.NULL)
            glib_loop.quit()
            glib_thread.join()

# --- Phần Producer cho Camera ToF (Dựa trên TCP) ---
def receive_all(sock, n):
    """Hàm helper để đảm bảo nhận đủ n bytes từ socket."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

async def run_tof_producer(frame_queue: asyncio.Queue, ip: str, port: int, stop_event: asyncio.Event):
    logger.info("Attempting to connect to ToF Slave...")
    # Kết nối TCP
    reader, writer = await asyncio.open_connection(ip, port)
    logger.info("Connection to ToF Slave successful!")

    try:
        while not stop_event.is_set():
            # Chạy hàm blocking trong một thread riêng
            tof_frames = await asyncio.to_thread(request_and_receive_tof_frames, reader, writer)
            if tof_frames:
                try:
                    frame_queue.put_nowait(tof_frames)
                except asyncio.QueueFull:
                    logger.warning("ToF frame queue is full, dropping frames.")
            await asyncio.sleep(0.05) # Điều chỉnh tần suất lấy frame
    except asyncio.CancelledError:
        logger.info("ToF Producer task cancelled.")
    finally:
        logger.info("Stopping ToF Producer...")
        writer.close()
        await writer.wait_closed()

def request_and_receive_tof_frames(reader, writer):
    """Hàm blocking để giao tiếp với slave, được chạy trong thread."""
    try:
        # Sử dụng API stream của asyncio
        writer.write(b"CAPTURE")
        # await writer.drain() # Không cần thiết nếu auto-draining

        header_len_bytes = reader.readexactly(4)
        header_len = struct.unpack('>I', header_len_bytes)[0]
        if header_len == 0: return None

        header_bytes = reader.readexactly(header_len)
        header = json.loads(header_bytes.decode('utf-8'))
        
        depth_size = header['depth_shape'][0] * header['depth_shape'][1] * np.dtype(header['depth_dtype']).itemsize
        depth_bytes = reader.readexactly(depth_size)
        depth_frame = np.frombuffer(depth_bytes, dtype=np.dtype(header['depth_dtype'])).reshape(header['depth_shape'])
        
        amp_size = header['amp_shape'][0] * header['amp_shape'][1] * np.dtype(header['amp_dtype']).itemsize
        amp_bytes = reader.readexactly(amp_size)
        amp_frame = np.frombuffer(amp_bytes, dtype=np.dtype(header['amp_dtype'])).reshape(header['amp_shape'])
        
        return depth_frame, amp_frame
    except Exception as e:
        logger.error(f"Error in request_and_receive_tof_frames: {e}")
        return None