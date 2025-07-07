# modules/stereo_camera_producer.py
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
import time
logger = get_logger(__name__)
Gst.init(None)

# --- Các hàm helper cho TCP (giữ nguyên từ main_app) ---
def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet) 
    return data

def request_and_receive_tof_frames(sock):
    try:
        sock.sendall(b"CAPTURE")
        # ... (Toàn bộ logic nhận frame ToF qua socket giữ nguyên) ...
        header_len_bytes = receive_all(sock, 4)
        if not header_len_bytes: return None, None, None
        header_len = struct.unpack('>I', header_len_bytes)[0]
        if header_len == 0: return None, None, None
        header_bytes = receive_all(sock, header_len)
        header = json.loads(header_bytes.decode('utf-8'))
        depth_size = header['depth_shape'][0] * header['depth_shape'][1] * np.dtype(header['depth_dtype']).itemsize
        depth_bytes = receive_all(sock, depth_size)
        depth_frame = np.frombuffer(depth_bytes, dtype=np.dtype(header['depth_dtype'])).reshape(header['depth_shape'])
        amp_size = header['amp_shape'][0] * header['amp_shape'][1] * np.dtype(header['amp_dtype']).itemsize
        amp_bytes = receive_all(sock, amp_size)
        amp_frame = np.frombuffer(amp_bytes, dtype=np.dtype(header['amp_dtype'])).reshape(header['amp_shape'])
        return depth_frame, amp_frame
    except Exception as e:
        logger.error(f"Error requesting/receiving ToF frames: {e}")
        return None, None

class StereoCameraProducer:
    def __init__(self, rgb_queue: asyncio.Queue, tof_queue: asyncio.Queue, slave_ip: str, slave_port: int, device='/dev/rgb_cam', width=640, height=480, framerate=15):
        self.loop = asyncio.get_event_loop()
        self.rgb_queue = rgb_queue
        self.tof_queue = tof_queue
        self.slave_socket = self._connect_to_slave(slave_ip, slave_port)
        
        if not self.slave_socket:
            raise ConnectionError("Failed to connect to ToF slave device.")

        self.pipeline = Gst.parse_launch(
            f"v4l2src device={device} ! image/jpeg, width={width}, height={height}, framerate={framerate}/1 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_new_rgb_sample)

    def _connect_to_slave(self, ip, port, max_retries=5, retry_delay=3):
        # ... (Hàm connect_to_slave từ main_app được chuyển vào đây) ...
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to Slave at {ip}:{port} (Attempt {attempt+1})...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(10.0)
                s.connect((ip, port))
                logger.info(">>> Connection to Slave successful! <<<")
                return s
            except Exception as e:
                logger.warning(f"Connection failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        return None

    def _on_new_rgb_sample(self, sink):
        """Callback được gọi mỗi khi có frame RGB mới từ GStreamer."""
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        rgb_frame = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.uint8).reshape((h, w, 3))
        
        # Có frame RGB, ngay lập tức yêu cầu frame ToF
        tof_depth, tof_amp = request_and_receive_tof_frames(self.slave_socket)

        if rgb_frame is not None and tof_depth is not None:
            try:
                # Đẩy cả 3 frame vào một gói dữ liệu duy nhất
                self.rgb_queue.put_nowait(rgb_frame)
                self.tof_queue.put_nowait((tof_depth, tof_amp))
                logger.debug("Successfully put synchronized stereo frame pair into queues.")
            except asyncio.QueueFull:
                logger.warning("A frame queue is full, dropping stereo pair.")
        
        return Gst.FlowReturn.OK

    async def run(self):
        logger.info("Starting Stereo Camera Producer...")
        self.pipeline.set_state(Gst.State.PLAYING)
        glib_loop = GLib.MainLoop()
        glib_thread = threading.Thread(target=glib_loop.run)
        glib_thread.start()
        
        try:
            await asyncio.Event().wait()
        finally:
            logger.info("Stopping Stereo Camera Producer...")
            self.pipeline.set_state(Gst.State.NULL)
            if self.slave_socket: self.slave_socket.close()
            glib_loop.quit()
            glib_thread.join()

async def start_stereo_producer(rgb_queue: asyncio.Queue, tof_queue: asyncio.Queue, ip: str, port: int, device: str):
    """Hàm khởi động producer."""
    try:
        producer = StereoCameraProducer(rgb_queue, tof_queue, ip, port, device)
        await producer.run()
    except asyncio.CancelledError:
        logger.info("Stereo Producer task cancelled.")
    except Exception as e:
        logger.error(f"Error in Stereo Producer: {e}", exc_info=True)