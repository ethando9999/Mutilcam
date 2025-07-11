import asyncio
import websockets
import json
import numpy as np
from datetime import datetime


def _numpy_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"{obj!r} is not JSON serializable")


class Socket_Sender:
    """
    Client class for sending packets to a WebSocket server.
    """

    def __init__(self, server_uri: str):
        self.server_uri = server_uri
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
        """
        try:
            message = json.dumps(packets, indent=2, default=_numpy_converter)

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


async def main():
    # Example usage
    id_queue = asyncio.Queue()
    server_uri = "ws://192.168.1.208:3000/ws"

    # Start background sender
    asyncio.create_task(start_id_sender(id_queue, server_uri))

    # Simulate enqueueing a packet
    example_packet = {
        "frame_id": 1001,
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

    await asyncio.sleep(2)  # Wait for packet to be processed



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[INFO] Expected error in simulation: {e}")
