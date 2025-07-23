# file: core/socket_sender.py

import asyncio
import json
import websockets
import numpy as np
from utils.logging_python_orangepi import get_logger
import uuid

logger = get_logger(__name__)

def _numpy_converter(obj):
    """
    Hàm helper để chuyển đổi các kiểu dữ liệu NumPy sang kiểu dữ liệu JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Đối tượng {obj!r} thuộc loại {type(obj).__name__} không thể chuyển đổi sang JSON")

class Socket_Sender:
    """
    Quản lý một kết nối WebSocket bền vững, mạnh mẽ.
    Tự động kết nối lại và xác minh kết nối trước mỗi lần gửi.
    """
    def __init__(self, server_uri: str):
        self.server_uri = server_uri
        self.websocket = None
        self.lock = asyncio.Lock()
        # Khởi tạo kết nối ngay khi tạo đối tượng
        self._connect_task = asyncio.create_task(self._ensure_connection())
        logger.info(f"Socket_Sender khởi tạo cho URI: {self.server_uri}")

    async def _ensure_connection(self):
        """
        Thử kết nối ban đầu và giữ lại kết nối.
        """
        while self.websocket is None:
            try:
                self.websocket = await asyncio.wait_for(
                    websockets.connect(self.server_uri), timeout=5.0
                )
                logger.info(f">>> Kết nối WebSocket thành công tới {self.server_uri}! <<<")
                return
            except Exception as e:
                logger.error(f"Kết nối tới {self.server_uri} thất bại: {e}. Thử lại sau 3 giây.")
                self.websocket = None
                await asyncio.sleep(3)

    async def send_packets(self, packets: dict):
        """
        Gửi dữ liệu một cách an toàn. Tự động xử lý mọi sự cố kết nối
        và sẽ thử lại cho đến khi gửi tin nhắn thành công.
        """
        message_to_send = json.dumps(packets, indent=2, default=_numpy_converter)

        async with self.lock:
            # Đảm bảo kết nối ban đầu đã hoàn thành
            if self._connect_task is not None:
                await self._connect_task
                self._connect_task = None

            while True:
                if self.websocket is None:
                    # Tái kết nối khi cần
                    await self._ensure_connection()
                    continue

                try:
                    # Kiểm tra kết nối
                    await asyncio.wait_for(self.websocket.ping(), timeout=2.0)
                    await self.websocket.send(message_to_send)
                    logger.info(f"✅ Đã gửi tới {self.server_uri}:\n{message_to_send}")
                    return
                except Exception as e:
                    logger.warning(
                        f"Kết nối tới {self.server_uri} có vấn đề ({type(e).__name__}). Đóng và tạo lại..."
                    )
                    try:
                        await self.websocket.close()
                    except Exception:
                        pass
                    self.websocket = None

async def start_socket_sender(socket_queue: asyncio.Queue, server_uri: str):
    """
    Worker lắng nghe hàng đợi và gửi gói tin đi.
    """
    sender = Socket_Sender(server_uri)
    
    while True:
        try:
            packet = await socket_queue.get()
            if packet is None:
                break
            
            asyncio.create_task(sender.send_packets(packet))
            socket_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Sender worker cho {server_uri} bị hủy.")
            break
        except Exception as e:
            logger.error(f"Lỗi không mong muốn trong sender worker cho {server_uri}: {e}", exc_info=True)