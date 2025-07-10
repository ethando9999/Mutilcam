import asyncio
import websockets
import json
from datetime import datetime

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


async def main():
    # Example usage
    client = Socket_Sender(server_uri="ws://192.168.1.208:3000/ws")

    # Example packet data
    example_packet = {
        "person_id": 123,
        "gender": "male",
        "race": "asian",
        "age": "Teenager",
        "time_detect": datetime.now().isoformat(),
        "camera_id": "001",
        "point3D": [450.5, 320.0, 1.8]
    }

    await client.send_packets(example_packet)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[INFO] Expected error in simulation: {e}")
