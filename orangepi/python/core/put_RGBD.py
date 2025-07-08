# file: python/core/put_RGBD.py (Phiên bản cuối cùng - Xử lý qua file)

import asyncio
import cv2
import numpy as np
import os
import socket
import json
import struct
import time
from typing import Tuple, Optional

from utils.logging_python_orangepi import get_logger
import config

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
    def __init__(self, rgb_camera_id: int):
        self.stop_event = asyncio.Event()
        self.rgb_camera_id = rgb_camera_id
        
        self.rgb_frame_dir = "frames/rgb"
        self.depth_frame_dir = "frames/depth"  # Thư mục lưu file depth .npy
        self.amp_frame_dir = "frames/amp"      # Thư mục lưu file amp .npy
        for d in [self.rgb_frame_dir, self.depth_frame_dir, self.amp_frame_dir]:
            os.makedirs(d, exist_ok=True)
            
        self.slave = SlaveCommunicator(
            slave_ip=config.OPI_CONFIG["slave_ip"],
            tcp_port=config.OPI_CONFIG["tcp_port"]
        )
        self.frame_count = 0
        logger.info("FramePutter (File-based) đã được khởi tạo.")

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(self.rgb_camera_id)
        if not cap.isOpened():
            logger.error(f"❌ Không mở được camera RGB với ID: {self.rgb_camera_id}")
            return

        logger.info(f"Bắt đầu vòng lặp Putter, camera ID: {self.rgb_camera_id}.")
        
        try:
            while not self.stop_event.is_set():
                if frame_queue.full():
                    await asyncio.sleep(0.1)
                    continue

                tof_task = self.slave.request_and_receive_tof_frames()
                rgb_task = loop.run_in_executor(None, cap.read)
                (depth_frame, amp_frame), (ret, rgb_frame) = await asyncio.gather(tof_task, rgb_task)

                if not ret or rgb_frame is None or depth_frame is None:
                    logger.warning("Không lấy đủ dữ liệu từ camera RGB hoặc ToF, bỏ qua chu trình này.")
                    await asyncio.sleep(0.5)
                    continue
                
                base_filename = f"capture_{time.time():.3f}_{self.frame_count:06d}"
                self.frame_count += 1
                
                # --- LƯU TẤT CẢ DỮ LIỆU RA FILE ---
                rgb_path = os.path.join(self.rgb_frame_dir, f"{base_filename}_rgb.jpg")
                depth_path = os.path.join(self.depth_frame_dir, f"{base_filename}_depth.npy")
                amp_path = os.path.join(self.amp_frame_dir, f"{base_filename}_amp.npy") if amp_frame is not None else None

                # Chạy các tác vụ I/O song song
                save_tasks = [
                    loop.run_in_executor(None, cv2.imwrite, rgb_path, rgb_frame)
                ]
                # Chỉ lưu depth và amp nếu chúng tồn tại
                if depth_frame is not None:
                    save_tasks.append(loop.run_in_executor(None, np.save, depth_path, depth_frame))
                if amp_frame is not None and amp_path is not None:
                    save_tasks.append(loop.run_in_executor(None, np.save, amp_path, amp_frame))
                
                await asyncio.gather(*save_tasks)
                
                # --- ĐƯA ĐƯỜNG DẪN VÀO QUEUE ---
                data_packet = (rgb_path, depth_path, amp_path)
                await frame_queue.put(data_packet)

        finally:
            logger.info("Dừng vòng lặp Putter, dọn dẹp tài nguyên...")
            cap.release()
            await self.slave.close()
            await frame_queue.put(None)

    async def stop(self):
        self.stop_event.set()

async def start_putter(frame_queue: asyncio.Queue, camera_id: int):
    logger.info("Khởi động Putter với luồng xử lý qua file...")
    putter = FramePutter(rgb_camera_id=camera_id)
    try:
        await putter.put_frames_queue(frame_queue)
    except asyncio.CancelledError:
        logger.info("Putter task was cancelled.")
    finally:
        await putter.stop()