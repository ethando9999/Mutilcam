# file: core/socket_sender.py (Phiên bản hoàn thiện - Xử lý kết nối mạnh mẽ)

import asyncio
import websockets
import json
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    """
    Quản lý kết nối WebSocket một cách mạnh mẽ. Tự động kết nối lại và 
    xác minh kết nối trước mỗi lần gửi để đảm bảo độ tin cậy.
    """
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.lock = asyncio.Lock()
        logger.info(f"WebSocketManager khởi tạo cho URI: {self.uri}")

    async def send_data(self, payload: dict):
        """
        Gửi dữ liệu một cách an toàn. Hàm này sẽ không thoát ra cho đến khi 
        gửi tin nhắn thành công, tự động xử lý mọi sự cố kết nối.
        """
        async with self.lock:
            while True:
                # BƯỚC 1: Nếu chưa có kết nối, hãy tạo nó.
                if self.websocket is None:
                    try:
                        # Cố gắng kết nối trong 5 giây
                        self.websocket = await asyncio.wait_for(websockets.connect(self.uri), timeout=5.0)
                        logger.info(f">>> Kết nối WebSocket thành công tới {self.uri}! <<<")
                    except Exception as e:
                        logger.error(f"Kết nối tới {self.uri} thất bại: {e}. Thử lại sau 3 giây.")
                        self.websocket = None # Đảm bảo reset khi lỗi
                        await asyncio.sleep(3)
                        continue # Quay lại đầu vòng lặp để thử kết nối lại

                # BƯỚC 2: Nếu đã có đối tượng kết nối, hãy thử sử dụng nó.
                try:
                    # Dùng ping() với timeout ngắn để kiểm tra sức khỏe kết nối.
                    # Đây là cách đáng tin cậy nhất.
                    await asyncio.wait_for(self.websocket.ping(), timeout=2.0)
                    
                    # Nếu ping thành công, gửi dữ liệu.
                    await self.websocket.send(json.dumps(payload))
                    logger.info(f"✅ Đã gửi tới {self.uri}: {json.dumps(payload)}")
                    
                    # Nếu gửi thành công, thoát khỏi vòng lặp và kết thúc.
                    return

                # BƯỚC 3: Nếu có bất kỳ lỗi nào ở BƯỚC 2, kết nối đã hỏng.
                except Exception as e:
                    logger.warning(f"Kết nối tới {self.uri} có vấn đề ({type(e).__name__}). Đóng kết nối cũ và tạo lại...")
                    try:
                        await self.websocket.close()
                    except:
                        # Bỏ qua lỗi khi đóng một kết nối đã chết
                        pass
                    
                    # Quan trọng: Reset về None để vòng lặp tiếp theo sẽ tạo kết nối mới ở BƯỚC 1.
                    self.websocket = None
                    # Không cần sleep, để vòng lặp thử lại ngay lập tức.


async def start_sender_worker(socket_queue: asyncio.Queue, opi_config: dict):
    """Worker này không thay đổi, nó chỉ việc gọi send_data."""
    height_uri = opi_config.get("SOCKET_HEIGHT_URI")
    count_uri = opi_config.get("SOCKET_COUNT_URI")
    
    if not height_uri or not count_uri:
        logger.error("URI cho WebSocket chưa được cấu hình. Worker sẽ không khởi động.")
        return

    height_sender = WebSocketManager(uri=height_uri)
    count_sender = WebSocketManager(uri=count_uri)

    logger.info("Khởi động worker gửi dữ liệu WebSocket (phiên bản ổn định nhất)...")

    while True:
        try:
            packet = await socket_queue.get()
            if packet is None:
                break

            packet_type = packet.get("type")
            data = packet.get("data")

            if packet_type == "height_data":
                asyncio.create_task(height_sender.send_data(data))
            elif packet_type == "person_count":
                asyncio.create_task(count_sender.send_data(data))
            
            socket_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Sender worker bị hủy.")
            break
        except Exception as e:
            logger.error(f"Lỗi không mong muốn trong sender worker: {e}", exc_info=True)