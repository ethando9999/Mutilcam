# file: python/core/put_RGBD_final.py

import asyncio
import cv2
import numpy as np
import socket
import json
import struct
import time
from typing import Tuple, Optional, Dict, Any

# Đảm bảo các đường dẫn import này đúng với cấu trúc dự án của bạn
from utils.logging_python_orangepi import get_logger
from core.lastest_queue import LatestFrameQueue
from utils.rmbg_mog import BackgroundRemover  
logger = get_logger(__name__)

class SlaveCommunicator:
    """
    Quản lý kết nối và giao tiếp TCP với Slave để nhận frame ToF (Depth + Amplitude).
    """
    def __init__(self, slave_ip: str, tcp_port: int, timeout: float = 10.0):
        self.slave_ip = slave_ip
        self.tcp_port = tcp_port
        self.timeout = timeout
        self.sock = None
        self.lock = asyncio.Lock()
        logger.info(f"SlaveCommunicator sẵn sàng kết nối tới {slave_ip}:{tcp_port}")

    async def _connect(self) -> bool:
        """Thiết lập kết nối socket mới."""
        try:
            logger.info(f"Đang kết nối tới Slave tại {self.slave_ip}:{self.tcp_port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            await asyncio.get_running_loop().sock_connect(self.sock, (self.slave_ip, self.tcp_port))
            logger.info(">>> Kết nối Slave thành công! <<<")
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.error(f"Kết nối Slave thất bại: {e}")
            if self.sock: self.sock.close()
            self.sock = None
            return False

    async def _recv_all(self, num_bytes: int) -> Optional[bytes]:
        """Nhận chính xác `num_bytes` từ socket một cách an toàn."""
        buffer = bytearray()
        loop = asyncio.get_running_loop()
        try:
            while len(buffer) < num_bytes:
                packet = await loop.sock_recv(self.sock, num_bytes - len(buffer))
                if not packet:
                    raise ConnectionError(f"Kết nối đã đóng, chỉ nhận được {len(buffer)}/{num_bytes} bytes.")
                buffer.extend(packet)
            return bytes(buffer)
        except (ConnectionError, socket.timeout, OSError) as e:
            logger.error(f"Lỗi khi nhận dữ liệu: {e}")
            return None

    async def close(self):
        """Đóng kết nối socket một cách an toàn."""
        async with self.lock:
            if self.sock:
                logger.info("Đang đóng kết nối với Slave...")
                self.sock.close()
                self.sock = None

    async def request_and_receive_tof_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Gửi lệnh và nhận cả depth và amplitude frame từ Slave."""
        async with self.lock:
            if not self.sock:
                if not await self._connect():
                    await asyncio.sleep(2)
                    return None, None
            
            try:
                loop = asyncio.get_running_loop()
                await loop.sock_sendall(self.sock, b"CAPTURE")

                header_len_bytes = await self._recv_all(4)
                if not header_len_bytes: raise ConnectionError("Không nhận được độ dài header.")
                
                header_len = struct.unpack('>I', header_len_bytes)[0]
                if header_len == 0: raise ConnectionError("Slave báo lỗi (header rỗng).")

                header_bytes = await self._recv_all(header_len)
                if not header_bytes: raise ConnectionError("Không nhận được header.")
                header = json.loads(header_bytes.decode('utf-8'))

                depth_shape = tuple(header['depth_shape'])
                depth_dtype = np.dtype(header['depth_dtype'])
                depth_size = int(np.prod(depth_shape) * depth_dtype.itemsize)
                depth_bytes = await self._recv_all(depth_size)
                if not depth_bytes: raise ConnectionError("Không nhận được dữ liệu depth.")
                depth_frame = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape)

                amp_frame = None
                if 'amp_shape' in header:
                    amp_shape = tuple(header['amp_shape'])
                    amp_dtype = np.dtype(header['amp_dtype'])
                    amp_size = int(np.prod(amp_shape) * amp_dtype.itemsize)
                    amp_bytes = await self._recv_all(amp_size)
                    if not amp_bytes: raise ConnectionError("Không nhận được dữ liệu amplitude.")
                    amp_frame = np.frombuffer(amp_bytes, dtype=amp_dtype).reshape(amp_shape)
                
                return depth_frame, amp_frame

            except Exception as e:
                logger.error(f"Giao tiếp Slave thất bại: {e}. Đóng kết nối để thử lại.")
                if self.sock: self.sock.close()
                self.sock = None
                return None, None

class FramePutter:
    """
    Lấy frame, thực hiện trừ nền, và đưa gói dữ liệu (RGB, Depth, Mask)
    vào hàng đợi trong RAM.
    """
    def __init__(self, config: Dict[str, Any]):
        self.stop_event = asyncio.Event()
        self.rgb_camera_id = config.get("rgb_camera_id", 0)
        self.target_fps = config.get("rgb_framerate", 15)
        self.frame_interval = 1.0 / self.target_fps
        
        # <<< TÍCH HỢP TRỪ NỀN >>>
        self.learning_time_sec = config.get("bg_learning_time", 3) # Thời gian học nền (giây)
        self.bg_remover = BackgroundRemover()
        # <<< KẾT THÚC TÍCH HỢP >>>

        slave_ip = config.get("slave_ip")
        tcp_port = config.get("tcp_port")
        if not slave_ip or not tcp_port:
            raise ValueError("Configuration for Putter must include 'slave_ip' and 'tcp_port'")
            
        self.slave = SlaveCommunicator(slave_ip=slave_ip, tcp_port=tcp_port)
        logger.info(f"FramePutter (With BgRemoval) đã khởi tạo với mục tiêu {self.target_fps} FPS.")

    async def _learn_background_phase(self, cap: cv2.VideoCapture):
        """
        Chạy trong vài giây đầu tiên để xây dựng mô hình nền tĩnh.
        """
        logger.info(f"Bắt đầu giai đoạn học nền trong {self.learning_time_sec} giây...")
        start_time = time.time()
        loop = asyncio.get_running_loop()

        frame_count = 0
        while time.time() - start_time < self.learning_time_sec:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            # Cập nhật nền bằng cách gọi hàm từ BackgroundRemover
            # Chạy trong executor vì đây là tác vụ CPU-bound
            await loop.run_in_executor(None, self.bg_remover.update_background, frame)
            frame_count += 1
        
        logger.info(f"✅ Hoàn tất giai đoạn học nền sau {frame_count} frames.")

    async def put_frames_queue(self, frame_queue: LatestFrameQueue):
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(self.rgb_camera_id)
        if not cap.isOpened():
            logger.error(f"❌ Không mở được camera RGB với ID: {self.rgb_camera_id}")
            return

        # --- GIAI ĐOẠN 1: HỌC NỀN ---
        await self._learn_background_phase(cap)

        logger.info(f"Bắt đầu vòng lặp Putter chính, camera ID: {self.rgb_camera_id}.")
        try:
            while not self.stop_event.is_set():
                loop_start_time = time.time()

                # --- GIAI ĐOẠN 2: CHẠY VÀ XỬ LÝ ---
                ret, rgb_frame = await loop.run_in_executor(None, cap.read)
                if not ret or rgb_frame is None:
                    logger.warning("Không lấy được frame từ camera RGB, thử lại...")
                    await asyncio.sleep(0.5)
                    continue

                depth_frame, _ = await self.slave.request_and_receive_tof_frames()
                if depth_frame is None:
                    logger.warning("Không nhận được frame từ Slave (ToF). Thử lại sau 2 giây...")
                    await asyncio.sleep(2.0)
                    continue

                # Áp dụng trừ nền để lấy mặt nạ tiền cảnh
                fgmask, _, _ = await loop.run_in_executor(None, self.bg_remover.remove_background, rgb_frame)

                # Đóng gói dữ liệu mới (thêm fgmask) và gửi đi
                data_packet = (rgb_frame, depth_frame, fgmask, time.time())
                await frame_queue.put(data_packet)

                # Điều tiết FPS
                elapsed_time = time.time() - loop_start_time
                sleep_time = self.frame_interval - elapsed_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            logger.info("Dừng vòng lặp Putter, dọn dẹp tài nguyên...")
            cap.release()
            await self.slave.close()

    async def stop(self):
        self.stop_event.set()


async def start_putter(frame_queue: LatestFrameQueue, putter_config: Dict[str, Any]):
    """
    Khởi động Putter với chức năng trừ nền.
    """
    logger.info("Khởi động Putter (có trừ nền)...")
    try:
        putter = FramePutter(config=putter_config)
        await putter.put_frames_queue(frame_queue)
    except asyncio.CancelledError:
        logger.info("Putter task was cancelled.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng không mong muốn trong start_putter: {e}", exc_info=True)
        raise
