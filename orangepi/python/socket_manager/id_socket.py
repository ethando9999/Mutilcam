import asyncio
from .socket_sender import Socket_Sender
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)


async def start_id_sender(id_socket_queue: asyncio.Queue, server_uri: str):
    """
    Start a background task that continuously sends packets from the queue.
    """
    sender = Socket_Sender(server_uri)
    while True:
        packets = await id_socket_queue.get()
        logger.info(f"[LOG] send packet to server: {packets}")
        response = await sender.send_packets(packets)
        logger.info(f"[LOG] Server response for packet: {response}")
        id_socket_queue.task_done()
