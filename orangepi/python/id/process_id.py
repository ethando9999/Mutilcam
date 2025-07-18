import os
import cv2

import time
import asyncio
from collections import Counter
from .managerID import ManagerID
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class ProcessID:
    """Orchestrates frame-by-frame processing and uses ManagerID to handle ReID."""
    def __init__(self, config: dict, queue: asyncio.Queue,  id_queue: asyncio.Queue):
        self.queue = queue
        self.config = config
        self.manager = ManagerID(id_queue, config['output_dir'], config)
        self.frame_index = 0
        self.stats = Counter()
        self.fps_avg = 0

    async def analyze(self, data: dict) -> str:
        try:
            start = time.time()
            img = data['human_box']
            parts = data.get('body_parts', {})
            feat_task = self.manager.extract_feature(img)
            tasks = [feat_task]

            head = parts.get('head')
            if head:
                x1, y1, x2, y2 = head
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        tasks.append(self.manager.extract_face(crop))

            results = await asyncio.gather(*tasks)
            feature = results[0]
            face_res = results[1] if len(results) > 1 else (None,) * 5
            face_emb = face_res[0]
            age = face_res[1] 
            gender = face_res[2] 
            race = face_res[4] 

            kwargs = dict(
                gender=gender,
                race=race,
                age=age,
                body_color=data.get('body_color'),
                feature_person=feature,
                face_embedding=face_emb,
                est_height_m=data.get('est_height_m'),
                head_point_3d=data.get('head_point_3d'),
                bbox=data.get('bbox'),
                frame_id=data.get('frame_id'), 
                map_keypoints=data.get('map_keypoints'),
                time_detect=data.get('time_detect')
            )

            person_id = await self.manager.assign_id(**kwargs)

            out_dir = os.path.join(self.config['output_dir'], f"id_{person_id}")
            os.makedirs(out_dir, exist_ok=True)
            filename = f"frame_{self.frame_index}.jpg"
            await asyncio.to_thread(cv2.imwrite, os.path.join(out_dir, filename), img)

            self.stats[person_id] += 1
            dur = time.time() - start
            fps = 1 / dur if dur > 0 else 0
            self.fps_avg = (self.fps_avg * (sum(self.stats.values()) - 1) + fps) / sum(self.stats.values())
            self.frame_index += 1
            logger.info(f"Processed frame for {person_id} - FPS_avg: {self.fps_avg:.2f}")
            return person_id
        except Exception as e:
            logger.exception(f"Exception during analyze(): {e}")
            return "error"


    async def run(self):
        while True:
            data = await self.queue.get()
            if data is None:
                break
            try:
                await self.analyze(data)
            except Exception as e:
                logger.error(f"Error: {e}")
            finally:
                self.queue.task_done()
        await self.manager.close()

# Usage example:
# config = DEFAULT_CONFIG
# proc = ProcessID(config)
# asyncio.run(proc.run())
