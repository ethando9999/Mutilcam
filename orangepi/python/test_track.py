import asyncio
from datetime import datetime, UTC

from track_local.byte_track import TrackingManager
from core.socket import start_socket_sender
import config


async def produce_dummy_detection(detection_queue: asyncio.Queue):
    packet = {
        "camera_id": "cam_1",
        "frame_id": 1,
        "time_detect": datetime.now().isoformat(),
        "people_list": [
            {
                "bbox": [10, 20, 100, 200, 0.9], 
                "world_point_xy": [1.0, 2.0],
                "attributes": {
                    "gender_analysis": {"gender": "male"},
                    "clothing_analysis": {
                        "classification": {
                            "sleeve_type": "short",
                            "pants_type": "jeans",
                            "skin_tone_bgr": [180, 150, 130]
                        },
                        "raw_color_data": {
                            "torso_colors": [{"bgr": [0, 0, 255], "percentage": 1.0}],
                            "thigh_colors": [{"bgr": [0, 255, 0], "percentage": 1.0}]
                        },
                    },
                },
                "est_height_m": 1.7,
            }
        ]
    }
    await detection_queue.put(packet)
    await detection_queue.put(None)  # để kết thúc tracker


async def main():
    detection_queue = asyncio.Queue()
    profile_queue = asyncio.Queue()
    all_tasks = []
    tracker = TrackingManager(detection_queue, profile_queue)
    all_tasks.append(asyncio.create_task(tracker.run())) 
    all_tasks.append(asyncio.create_task(start_socket_sender(profile_queue, config.OPI_CONFIG["SOCKET_TRACK_COLOR_URI"])))

    await produce_dummy_detection(detection_queue)

    await asyncio.sleep(1)  # chờ tracker đẩy xong qua queue
    await profile_queue.put(None)  # kết thúc SocketSender

    await asyncio.gather(*all_tasks)


if __name__ == "__main__":
    asyncio.run(main())
