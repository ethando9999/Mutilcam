# file: python/core/socket_sender.py (Đã thêm log payload)

import asyncio
import json
import uuid
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

def _numpy_converter(obj):
    """
    Hàm helper để chuyển đổi các kiểu dữ liệu không thể tuần tự hóa của NumPy và UUID
    sang các kiểu dữ liệu gốc của Python mà JSON có thể hiểu được.
    """
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, uuid.UUID): return str(obj)
    raise TypeError(f"Đối tượng {obj!r} thuộc loại {type(obj).__name__} không thể chuyển đổi sang JSON")


class SocketSender:
    """
    Một worker quản lý một kết nối WebSocket bền vững và mạnh mẽ.
    - Lắng nghe dữ liệu từ một hàng đợi (asyncio.Queue).
    - Tự động kết nối lại khi mất kết nối (Retry Connect).
    - Ghi log lỗi kết nối một cách tinh gọn và ghi log payload gửi đi.
    """
    def __init__(self, uri: str, data_queue: asyncio.Queue, name: str = "SocketSender", retry_delay: int = 3):
        self.name = name
        self.data_queue = data_queue
        self.retry_delay = retry_delay
        self.uri = self._sanitize_uri(uri)
        logger.info(f"[{self.name}] Khởi tạo worker cho URI: {self.uri}")

    def _sanitize_uri(self, uri: str) -> str:
        """Kiểm tra và sửa scheme của URI nếu cần thiết."""
        if uri.startswith("https://"):
            logger.warning(f"[{self.name}] URI bắt đầu bằng 'https://'. Tự động đổi thành 'wss://'.")
            return uri.replace("https://", "wss://", 1)
        if uri.startswith("http://"):
            logger.warning(f"[{self.name}] URI bắt đầu bằng 'http://'. Tự động đổi thành 'ws://'.")
            return uri.replace("http://", "ws://", 1)
        if not (uri.startswith("ws://") or uri.startswith("wss://")):
             raise ValueError(f"[{self.name}] URI '{uri}' không hợp lệ. Phải bắt đầu bằng ws:// hoặc wss://.")
        return uri

    async def run(self):
        """
        Vòng lặp chính của worker: kết nối và gửi dữ liệu từ hàng đợi.
        """
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=20, ping_timeout=20) as websocket:
                    logger.info(f"[{self.name}] ✅ Kết nối WebSocket thành công tới {self.uri}")
                    while True:
                        packet = await self.data_queue.get()

                        # ← Thêm check ở đây
                        if packet is None:
                            logger.info(f"[{self.name}] 🔌 Shutdown signal received. Exiting sender loop.")
                            self.data_queue.task_done()
                            return

                        try:
                            message_to_send = json.dumps(packet, default=_numpy_converter)
                            logger.info(f"[{self.name}] >> Gửi packet: {message_to_send}")
                            print(f"[{self.name}] >> Gửi packet: {message_to_send}")
                            await websocket.send(message_to_send)
                        except TypeError as e:
                            logger.error(f"[{self.name}] Lỗi chuyển đổi JSON: {e}. Bỏ qua packet.")
                        finally:
                            self.data_queue.task_done()

            except (ConnectionRefusedError, ConnectionClosed, asyncio.TimeoutError, InvalidStatus) as e:
                logger.warning(f"[{self.name}] ❌ Lỗi kết nối tới {self.uri}: {e}. Sẽ thử lại sau {self.retry_delay} giây.")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Lỗi không mong muốn: {e}. Sẽ thử lại sau {self.retry_delay} giây.", exc_info=True)
            
            await asyncio.sleep(self.retry_delay)