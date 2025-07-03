from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)

import asyncio
import json
import uuid
import random
from typing import Any, Callable, Dict, Optional

import aio_pika

from config import RABBITMQ_URL

class RabbitMQManager:
    """
    RabbitMQManager hỗ trợ remote-request non-blocking giữa các thiết bị:
      - Gửi request lookup tới exchange 'remote_requests_exchange'
      - Nhận request và trả lời ngay qua reply_to
      - Gửi/nhận responses, kèm future để chờ kết quả không blocking
    """

    def __init__(
        self,
        device_id: str,
        
        # QoS settings
        prefetch: int = 5,
        heartbeat: int = 120,
        retry_max: int = 5,
    ):
        self.url = RABBITMQ_URL
        self.device_id = device_id
        self.prefetch = prefetch
        self.heartbeat = heartbeat
        self.retry_max = retry_max

        # Connection objects
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None

        # Exchanges & queues
        self.remote_requests_exchange: Optional[aio_pika.Exchange] = None
        self.remote_requests_queue: Optional[aio_pika.Queue] = None
        self.response_queue: Optional[aio_pika.Queue] = None

        # Pending remote lookup futures
        self._pending: Dict[str, asyncio.Future] = {}

        # Callback to handle incoming requests
        self.on_request: Optional[Callable[[Dict[str, Any]], Any]] = None

        # Ready event
        self._ready = asyncio.Event()

        self._connected = False 

    async def start(self):
        while True:
            await self._connect()
            if not self._connected:
                logger.warning("[%s] Initial connect fail—retry in 5s", self.device_id)
                await asyncio.sleep(5)
                continue

            # only when truly connected
            self._ready.set()

            try:
                await asyncio.gather(
                    self._consume_requests(),
                    self._consume_responses(),
                )
            except asyncio.CancelledError:
                await self.stop()
                break
            except Exception as e:
                logger.error("[%s] Consumer error: %s", self.device_id, e)
            finally:
                # ← dù consumer exit vì bất kỳ lý do gì (error or normal), ta clear ready
                self._ready.clear()
                self._connected = False
                logger.info("[%s] RabbitMQ disconnected, will retry in 5s", self.device_id)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Đóng kết nối RabbitMQ"""
        if self.channel and not self.channel.is_closed:
            await self.channel.close()
        if self.connection and not self.connection.is_closed:
            await self.connection.close()

    async def publish_remote_request(
        self,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> asyncio.Future:
        """
        Gửi remote lookup request và trả về future để chờ kết quả match_id.
        Cho phép dùng temp_id làm request_id để dễ tracking.

        Args:
            payload: feature/body_color/metadata
            request_id: nếu đã có (e.g. temp_id) thì tái sử dụng
        Returns:
            Future chứa matched_id hoặc None
        """
        await self._ready.wait()
        # sử dụng request_id đã cho, nếu không có thì generate mới
        rid = str(request_id) if request_id is not None else str(uuid.uuid4())
        payload_full = {
            **payload,
            "request_id": rid,
            "device_id": self.device_id,
        }
        future = asyncio.get_event_loop().create_future()
        # ghi đè/đăng ký future với key rid
        self._pending[rid] = future

        message = aio_pika.Message(
            body=json.dumps(payload_full, default=_json_numpy_fallback).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            reply_to=self.response_queue.name,
        )
        await self.remote_requests_exchange.publish(message, routing_key="")
        logger.info("[%s] Published remote_request %s", self.device_id, rid)
        return future

    async def _connect(self) -> None:
        """Tạo kết nối robust, channel, exchange/queue cần thiết"""
        for attempt in range(1, self.retry_max + 1):
            try:
                if self.connection and not self.connection.is_closed:
                    await self.connection.close()

                self.connection = await aio_pika.connect_robust(
                    self.url, heartbeat=self.heartbeat
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=self.prefetch)

                # Exchange fanout cho remote requests
                self.remote_requests_exchange = await self.channel.declare_exchange(
                    "remote_requests_exchange", aio_pika.ExchangeType.FANOUT, durable=True
                )

                # Queue nhận requests cho thiết bị này
                queue_name = f"remote_requests_{self.device_id}"
                self.remote_requests_queue = await self.channel.declare_queue(
                    queue_name, durable=True
                )
                await self.remote_requests_queue.bind(
                    self.remote_requests_exchange
                )

                # Queue response tạm để nhận trả lời
                self.response_queue = await self.channel.declare_queue( 
                    f"response_{self.device_id}_{uuid.uuid4()}",
                    durable=False,
                    exclusive=True,
                    auto_delete=True,
                )
                self._connected = True
                logger.info("[%s] RabbitMQ connected", self.device_id)
                return
            except Exception as e:
                logger.error("[%s] RabbitMQ connect fail %s/%s: %s", self.device_id, attempt, self.retry_max, e)
                await asyncio.sleep(2 ** attempt)

        self._connected = False  # Sau khi thử hết mà vẫn fail

    async def _consume_requests(self) -> None:
        """Consumer cho các remote requests đến, trả kết quả qua reply_to"""
        await self._ready.wait()
        async with self.remote_requests_queue.iterator() as itr:
            async for message in itr:
                async with message.process():
                    try:
                        data = json.loads(message.body)
                        # Bỏ qua request từ chính mình
                        if data.get("device_id") == self.device_id:
                            continue
                        if not self.on_request:
                            continue
                        matched = await self.on_request(data)
                        # Trả kết quả về
                        response = {
                            "request_id": data["request_id"],
                            "matched_id": matched,
                            "device_id": self.device_id,
                        }
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps(response).encode(),
                                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            ),
                            routing_key=message.reply_to,
                        )
                        logger.warning("[%s] Replied request %s -> %s", self.device_id, data["request_id"], matched)

                    except Exception as ex:
                        logger.exception("Error handling remote request: %s", ex)

    async def _consume_responses(self) -> None:
        """Consumer cho các response trả về, resolve future tương ứng"""
        await self._ready.wait()
        async with self.response_queue.iterator() as itr:
            async for message in itr:
                async with message.process():
                    try:
                        data = json.loads(message.body)
                        rid = data.get("request_id")
                        fut = self._pending.pop(rid, None)
                        if fut and not fut.done():
                            fut.set_result(data.get("matched_id"))
                            logger.debug("[%s] Received response %s -> %s", self.device_id, rid, data.get("matched_id"))
                    except Exception as ex:
                        logger.exception("Error processing response: %s", ex)

import numpy as np 
def _json_numpy_fallback(obj):
    # NumPy scalar → Python scalar
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")
