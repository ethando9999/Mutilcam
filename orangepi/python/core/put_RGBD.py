# file: python/core/put_RGBD.py (v17 - Tối ưu và Hoàn chỉnh)
import asyncio
import cv2
import numpy as np
import os
import socket
import json
import struct
import time
from typing import Tuple, Optional

# Giả sử các import này đã tồn tại và đúng
from utils.logging_python_orangepi import get_logger
import config

logger = get_logger(__name__)

class SlaveCommunicator:
    """
    Quản lý kết nối và giao tiếp TCP (Yêu cầu/Phản hồi) với Slave 
    để nhận frame ToF một cách an toàn và bất đồng bộ.
    """
    def __init__(self, slave_ip: str, tcp_port: int, timeout: float = 5.0):
        self.slave_ip = slave_ip
        self.tcp_port = tcp_port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.lock = asyncio.Lock() # Sử dụng lock của asyncio để đảm bảo an toàn
        logger.info(f"SlaveCommunicator (Tối ưu) sẵn sàng kết nối tới {slave_ip}:{tcp_port}")

    async def _connect(self) -> bool:
        """Thực hiện kết nối socket. Trả về True nếu thành công."""
        try:
            logger.info(f"Đang thử kết nối tới Slave tại {self.slave_ip}:{self.tcp_port}...")
            loop = asyncio.get_running_loop()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(self.timeout)
            await loop.run_in_executor(None, self.sock.connect, (self.slave_ip, self.tcp_port))
            logger.info(">>> Kết nối Slave thành công! <<<")
            return True
        except Exception as e:
            logger.error(f"Kết nối Slave thất bại: {e}")
            if self.sock: self.sock.close()
            self.sock = None
            return False

    async def _recv_all_async(self, num_bytes: int) -> bytes:
        """Nhận chính xác `num_bytes` từ socket."""
        loop = asyncio.get_running_loop()
        buffer = bytearray()
        while len(buffer) < num_bytes:
            packet = await loop.run_in_executor(None, self.sock.recv, num_bytes - len(buffer))
            if not packet:
                raise ConnectionError("Kết nối đã đóng bất ngờ khi đang nhận dữ liệu.")
            buffer.extend(packet)
        return bytes(buffer)

    async def close(self):
        """Đóng kết nối socket một cách an toàn."""
        async with self.lock:
            if self.sock:
                logger.info("Đang đóng kết nối với Slave...")
                self.sock.close()
                self.sock = None

    async def request_and_get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Gửi lệnh và nhận cả depth và amplitude frame từ Slave."""
        async with self.lock:
            # Tự động kết nối lại nếu socket chưa được tạo hoặc đã bị đóng
            if self.sock is None:
                if not await self._connect():
                    await asyncio.sleep(2) # Chờ 2 giây trước khi thử lại
                    return None, None
            
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.sock.sendall, b'G')

                header_len_bytes = await self._recv_all_async(4)
                header_len = struct.unpack('>I', header_len_bytes)[0]
                
                if header_len == 0:
                    logger.warning("Slave báo lỗi (không có frame).")
                    return None, None

                header_bytes = await self._recv_all_async(header_len)
                header = json.loads(header_bytes.decode('utf-8'))

                depth_frame = None
                if 'depth_shape' in header and header['depth_shape']:
                    depth_shape = tuple(header['depth_shape'])
                    depth_dtype = np.dtype(header['depth_dtype'])
                    depth_size = int(np.prod(depth_shape) * depth_dtype.itemsize)
                    depth_bytes = await self._recv_all_async(depth_size)
                    depth_frame = np.frombuffer(depth_bytes, dtype=depth_dtype).reshape(depth_shape)

                # TODO: Thêm logic xử lý amplitude frame nếu Slave có gửi
                amp_frame = None
                
                return depth_frame, amp_frame

            except Exception as e:
                logger.error(f"Giao tiếp với Slave thất bại: {e}. Đang đóng kết nối để thử lại.")
                if self.sock: self.sock.close()
                self.sock = None # Đặt lại để lần gọi sau sẽ kết nối lại
                return None, None

class FramePutter:
    """
    Producer hiệu năng cao, chạy bất đồng bộ để lấy, lưu frame và đẩy vào hàng đợi.
    """
    def __init__(self, rgb_camera_id: int):
        self.stop_event = asyncio.Event()
        self.rgb_camera_id = rgb_camera_id
        
        self.rgb_frame_dir = "frames/rgb"
        self.depth_frame_dir = "frames/depth"
        self.amp_frame_dir = "frames/amp"
        for d in [self.rgb_frame_dir, self.depth_frame_dir, self.amp_frame_dir]:
            os.makedirs(d, exist_ok=True)
            
        self.slave = SlaveCommunicator(
            slave_ip=config.OPI_CONFIG["slave_ip"],
            tcp_port=config.OPI_CONFIG["tcp_port"]
        )
        self.frame_count = 0
        logger.info("FramePutter (Tối ưu) đã được khởi tạo.")

    async def put_frames_queue(self, frame_queue: asyncio.Queue):
        loop = asyncio.get_running_loop()
        cap = cv2.VideoCapture(self.rgb_camera_id) 
        if not cap.isOpened():
            logger.error(f"❌ Không mở được camera RGB với ID: {self.rgb_camera_id}")
            return

        logger.info(f"Bắt đầu vòng lặp Putter, camera ID: {self.rgb_camera_id}.")
        
        try:
            while not self.stop_event.is_set():
                # Xử lý backpressure: Nếu queue đầy, chờ một chút
                if frame_queue.full():
                    logger.warning("Hàng đợi frame đã đầy. Consumer đang xử lý chậm. Tạm dừng Putter...")
                    await asyncio.sleep(0.1) 
                    continue

                # Thực hiện lấy ảnh từ các camera đồng thời để đồng bộ tốt nhất
                tof_task = self.slave.request_and_get_frame()
                rgb_task = loop.run_in_executor(None, cap.read)
                
                # Chờ cả hai tác vụ hoàn thành
                (depth_frame, amp_frame), (ret, rgb_frame) = await asyncio.gather(tof_task, rgb_task)

                if not ret or rgb_frame is None or depth_frame is None:
                    logger.warning("Không lấy đủ dữ liệu từ camera RGB hoặc ToF, bỏ qua chu trình này.")
                    await asyncio.sleep(0.5) # Chờ một chút nếu có lỗi
                    continue
                
                base_filename = f"capture_{time.time():.3f}_{self.frame_count:06d}"
                self.frame_count += 1
                
                rgb_path = os.path.join(self.rgb_frame_dir, f"{base_filename}_rgb.jpg")
                depth_path = os.path.join(self.depth_frame_dir, f"{base_filename}_depth.npy")
                amp_path = os.path.join(self.amp_frame_dir, f"{base_filename}_amp.npy") if amp_frame is not None else None

                # Lưu các file song song để tối ưu I/O
                save_tasks = [
                    loop.run_in_executor(None, cv2.imwrite, rgb_path, rgb_frame),
                    loop.run_in_executor(None, np.save, depth_path, depth_frame)
                ]
                if amp_frame is not None and amp_path is not None:
                    save_tasks.append(loop.run_in_executor(None, np.save, amp_path, amp_frame))
                
                await asyncio.gather(*save_tasks)
                
                data_packet = (rgb_path, depth_path, amp_path)
                await frame_queue.put(data_packet)

        finally:
            logger.info("Dừng vòng lặp Putter, dọn dẹp tài nguyên...")
            cap.release()
            await self.slave.close()
            # Gửi tín hiệu kết thúc cho consumer
            await frame_queue.put(None)

    async def stop(self):
        """Kích hoạt sự kiện để dừng vòng lặp."""
        self.stop_event.set()

async def start_putter(frame_queue: asyncio.Queue, camera_id: int):
    """Hàm bao bọc (wrapper) để khởi tạo và chạy Putter."""
    logger.info("Khởi động Putter...")
    # Loại bỏ target_fps khỏi khởi tạo FramePutter
    putter = FramePutter(rgb_camera_id=camera_id)
    try:
        await putter.put_frames_queue(frame_queue)
    except asyncio.CancelledError:
        logger.info("Putter task đã bị hủy.")
    finally:
        await putter.stop() 