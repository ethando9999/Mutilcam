import asyncio
import websockets
import json
from datetime import datetime
from asyncio import Queue

class Socket_Sender:
    """
    Client class for sending packets to a WebSocket server.
    """

    def __init__(self, server_uri: str):
        """
        Args:
            server_uri (str): The WebSocket server URI (e.g., ws://host:port/ws).
        """
        self.server_uri = server_uri
        # Check connection during initialization
        asyncio.create_task(self._test_connection())

    async def _test_connection(self):
        try:
            async with websockets.connect(self.server_uri) as ws:
                print(f"[INFO] Successfully connected to {self.server_uri}")
        except Exception as e:
            print(f"[WARN] Failed to connect during init: {e}")

    async def send_packets(self, packets: dict) -> str:
        """
        Send a JSON packet to the WebSocket server and return the response.

        Args:
            packets (dict): The data to be sent as a JSON payload.

        Returns:
            str: The response from the WebSocket server.
        """
        message = json.dumps(packets, indent=2)

        try:
            async with websockets.connect(self.server_uri) as ws:
                await ws.send(message)
                print(f"[SENT] {message}")
                response = await ws.recv()
                print(f"[RECV] {response}")
                return response
        except ConnectionRefusedError:
            error_msg = f"[ERROR] Unable to connect to {self.server_uri}. Ensure the server is running."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[ERROR] Unexpected error: {e}"
            print(error_msg)
            return error_msg


async def start_id_sender(id_queue: Queue, server_uri: str):
    """
    Start a background task that continuously sends packets from the queue.

    Args:
        id_queue (Queue): An asyncio queue containing JSON-serializable packets.
        server_uri (str): The WebSocket server URI.
    """
    sender = Socket_Sender(server_uri)
    while True:
        packets = await id_queue.get()
        response = await sender.send_packets(packets)
        print(f"[LOG] Server response for packet: {response}")
        id_queue.task_done()


async def enqueue_packets(id_queue: Queue):
    """
    Continuously generate and enqueue example packets every 1 second.
    """
    frame_id = 1000
    while True:
        example_packet = {
            "frame_id": frame_id,
            "person_id": 1234,
            "gender": "male",
            "race": "asian",
            "age": "Adult",
            "height": 1.72,
            "time_detect": datetime.now().isoformat(),
            "camera_id": "001",
            "point3D": [450.5, 320.0, 1.8]
        }
        await id_queue.put(example_packet)
        print(f"[ENQUEUE] Packet for frame_id={frame_id} enqueued")
        frame_id += 1
        await asyncio.sleep(1)


async def main():
    # Example usage
    id_queue = asyncio.Queue()
    server_uri = "ws://192.168.1.108:9090/api/ws/camera"

    # Start background sender and enqueuer
    asyncio.create_task(start_id_sender(id_queue, server_uri))
    asyncio.create_task(enqueue_packets(id_queue))

    await asyncio.sleep(10)  # Run for 10 seconds (or adjust as needed)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[INFO] Expected error in simulation: {e}")
