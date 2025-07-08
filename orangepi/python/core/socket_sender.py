# file: core/websocket_sender.py (Module Gửi Dữ liệu qua WebSocket - Sửa lỗi cuối cùng)

import asyncio
import websockets
import json
from utils.logging_python_orangepi import get_logger
import config

logger = get_logger(__name__)

class WebSocketManager:
    """
    Quản lý một kết nối WebSocket duy nhất, bao gồm kết nối, gửi và tự động kết nối lại.
    """
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.lock = asyncio.Lock()
        logger.info(f"WebSocketManager khởi tạo cho URI: {self.uri}")

    async def _connect_if_needed(self):
        """Chỉ thực hiện kết nối nếu self.websocket chưa được thiết lập."""
        if self.websocket:
            return

        async with self.lock:
            # Kiểm tra lại sau khi chiếm được lock
            if self.websocket:
                return

            logger.info(f"Đang cố gắng kết nối tới WebSocket Server tại {self.uri}...")
            while True:
                try:
                    # Đặt timeout cho việc kết nối
                    self.websocket = await asyncio.wait_for(websockets.connect(self.uri), timeout=5.0)
                    logger.info(f">>> Kết nối WebSocket thành công tới {self.uri}! <<<")
                    break
                except Exception as e:
                    logger.error(f"Kết nối WebSocket tới {self.uri} thất bại: {e}. Thử lại sau 5 giây.")
                    self.websocket = None # Đảm bảo reset khi thất bại
                    await asyncio.sleep(5)

    async def send_data(self, payload: dict):
        """
        Gửi một payload JSON đến server. Tự động kết nối lại nếu cần.
        """
        try:
            # Luôn đảm bảo có một đối tượng kết nối
            await self._connect_if_needed()
            
            # Gửi dữ liệu
            await self.websocket.send(json.dumps(payload))
            logger.info(f"✅ Đã gửi tới {self.uri}: {json.dumps(payload)}")

            # Có thể đợi phản hồi nếu cần, nhưng nên có timeout
            # response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
            # logger.info(f"↩️ Phản hồi từ {self.uri}: {response}")

        except Exception as e:
            # Bất kỳ lỗi nào xảy ra (gửi thất bại, mất kết nối) đều được xử lý ở đây
            logger.error(f"Lỗi khi gửi/nhận dữ liệu tại {self.uri}: {e}. Sẽ đóng và kết nối lại lần sau.")
            # Đóng kết nối cũ (nếu có) và reset để lần gọi sau sẽ kết nối lại
            if self.websocket:
                try:
                    await self.websocket.close()
                except:
                    pass
            self.websocket = None


async def start_sender_worker(processing_queue: asyncio.Queue, opi_config: dict):
    """
    Worker lắng nghe hàng đợi, phân loại tin nhắn và gửi đến đúng WebSocket server.
    """
    height_uri = opi_config.get("SOCKET_HEIGHT_URI")
    count_uri = opi_config.get("SOCKET_COUNT_URI")
    table_id = opi_config.get("SOCKET_TABLE_ID")

    if not height_uri or not count_uri:
        logger.error("URI cho WebSocket chiều cao hoặc số lượng chưa được cấu hình. Worker sẽ không khởi động.")
        return

    height_sender = WebSocketManager(uri=height_uri)
    count_sender = WebSocketManager(uri=count_uri)

    logger.info("Khởi động worker gửi dữ liệu WebSocket đa luồng...")

    while True:
        try:
            packet = await processing_queue.get()
            if packet is None:
                logger.info("Nhận tín hiệu dừng, đóng worker gửi dữ liệu.")
                break

            packet_type = packet.get("type")
            data = packet.get("data")

            if packet_type == "height_data":
                height_m = data.get("est_height_m")
                if not height_m:
                    continue

                height_cm = height_m * 100.0
                if 150.0 <= height_cm <= 190.0:
                    payload = {
                        "table_id": table_id,
                        "height_cm": round(height_cm, 2)
                    }
                    asyncio.create_task(height_sender.send_data(payload))
                else:
                    logger.warning(f"Chiều cao {height_cm:.2f} cm nằm ngoài khoảng hợp lệ (150-190). Bỏ qua.")

            elif packet_type == "person_count":
                asyncio.create_task(count_sender.send_data(data))

            processing_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Sender worker bị hủy.")
            break
        except Exception as e:
            logger.error(f"Lỗi không mong muốn trong sender worker: {e}", exc_info=True)