import os
import cv2
import numpy as np
import time
import asyncio
from collections import Counter, defaultdict
import uuid
import shutil
import subprocess
from typing import Optional

from feature.feature_osnet import FeatureModel
from .reid_manager import ReIDManager
from face_processing.face_v2.face_analyze_new import FaceAnalyze
from rabbitMQ.rabbitMQ_manager import RabbitMQManager
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

from config import OPI_CONFIG

def has_enough_valid_keypoints(keypoints, min_valid_keypoints=14):
    if not keypoints or not isinstance(keypoints, (list, tuple)):
        return False
    return sum(1 for x, y in keypoints if x != 0 and y != 0) >= min_valid_keypoints

class ManagerID:
    """Handles ReID operations, DB, feature and face extraction, and remote lookups."""

    def __init__(self, id_queue: asyncio.Queue, output_dir: str, config: dict):
        self.id_queue = id_queue
        self.output_dir = output_dir 
        self.config = OPI_CONFIG
        self.device_id = OPI_CONFIG.get('device_id', str(uuid.uuid4()))

        # RabbitMQ setup
        self.rabbit = RabbitMQManager(device_id=self.device_id)
        self.rabbit.on_request = self._handle_remote_request
        self._rabbit_task = asyncio.create_task(self.rabbit.start())

        # Feature and face models
        self.feature_extractor = FeatureModel()
        self.face_analyze = FaceAnalyze()

        # ReID manager
        self.reid_manager = ReIDManager(
            db_path=config['db_path'],
            feature_threshold=config['feature_threshold'],
            color_threshold=config['color_threshold'],
            avg_threshold=config['avg_threshold'],
            top_k=config['top_k'],
            thigh_weight=config['thigh_weight'],
            torso_weight=config['torso_weight'],
            feature_weight=config['feature_weight'],
            color_weight=config['color_weight'],
        )

        self.merge_threshold = config['merge_threshold']
        self.temp_timeout = config['temp_timeout']
        self.min_detections = config['min_detections']

        # State
        self.remote_requested = set()
        self.temp_counts = defaultdict(int)
        self.timeout_tasks = {}
        self.temp_id = set()
        self.stats_lock = asyncio.Lock()

    async def extract_feature(self, image: np.ndarray) -> np.ndarray:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.feature_extractor.extract_feature, image
        )

    async def extract_face(self, crop: np.ndarray):
        return await self.face_analyze.analyze(crop)

    async def _handle_remote_request(self, data: dict):
        """Handle incoming remote ReID request."""
        try:
            arr = lambda k: np.array(data[k], dtype=np.float32) if data.get(k) is not None else None
            return await self.reid_manager.match_id(
                new_gender=data.get('gender'),
                new_race=data.get('race'),
                new_age=data.get('age'),
                new_body_color=arr('body_color'),
                new_feature=arr('feature'),
                new_face_embedding=arr('face_embedding'),
            )
        except Exception:
            logger.exception("Error handling remote request")
            return None

    async def _find_and_merge(self, temp_id: str) -> Optional[str]:
        """
        Try to find a closest match and merge.
        Return closest_id if merged, else None.
        """
        try:
            closest_id, score = await self.reid_manager.find_closest_by_id(temp_id)
            if closest_id and score >= self.merge_threshold:
                await self.reid_manager.merge_temp_person_id(temp_id, closest_id)
                self._move_folder(temp_id, closest_id)
                return closest_id
        except Exception:
            logger.exception(f"Error in merging temp {temp_id}")
        return None

    def _move_folder(self, src_id: str, dst_id: str):
        """Move output folder of src_id under dst_id."""
        src = os.path.join(self.output_dir, f"id_{src_id}")
        dst = os.path.join(self.output_dir, f"id_{dst_id}", f"id_{src_id}")
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(["mv", src, dst], check=False)
            shutil.rmtree(src, ignore_errors=True)

    async def handle_temp_update(self, person_id: str, normalized: dict, keypoints: Optional[dict]):
        self.temp_counts[person_id] += 1

        # 1) Merge nếu có thể
        merged = await self._find_and_merge(person_id)
        if merged:
            # nếu merge thành công thì không cần track temp_id nữa
            self.temp_id.discard(person_id)
            await self.reid_manager.update_person(merged, **normalized)
            return

        # 2) Nếu chưa merge: cập nhật và đảm bảo đã add vào temp set
        await self.reid_manager.update_person(person_id, **normalized)
        self.temp_id.add(person_id)  # <-- thêm vào đây

        # 3) Quyết định gửi remote hay timeout
        if has_enough_valid_keypoints(keypoints):
            await self._publish_remote(person_id, **normalized)
        else:
            await self._schedule_timeout(person_id)

    async def _publish_remote(self,
                               temp_id: str,
                               gender, race, age,
                               body_color, feature, face_embedding):
        """Send data to remote ReID and handle response."""
        try:
            await asyncio.wait_for(self.rabbit._ready.wait(), timeout=1)
            self.remote_requested.add(temp_id)
            payload = {"gender": gender, "race": race, "age": age,
                       "body_color": body_color, "feature": feature,
                       "face_embedding": face_embedding}
            fut = await self.rabbit.publish_remote_request(payload, request_id=str(temp_id))

            async def on_response():
                try:
                    matched = await asyncio.wait_for(fut, timeout=20)
                    if matched and matched != temp_id:
                        await self.reid_manager.replace_temp_person_id(temp_id, matched)
                except asyncio.TimeoutError:
                    logger.warning(f"Remote timeout for {temp_id}")
                except Exception:
                    logger.exception("Error handling remote response")

            asyncio.create_task(on_response())
        except Exception:
            logger.exception(f"Error publishing remote for {temp_id}")

    async def _schedule_timeout(self, temp_id: str):
        if temp_id in self.timeout_tasks:
            self.timeout_tasks[temp_id].cancel()

        async def watcher():
            try:
                await asyncio.sleep(self.temp_timeout)
                if (temp_id not in self.remote_requested
                        and self.temp_counts[temp_id] < self.min_detections):
                    # xóa hẳn
                    self.temp_id.discard(temp_id)
                    self.remote_requested.discard(temp_id)
                    self.temp_counts.pop(temp_id, None)
                    await self.reid_manager.remove_low_person_ids([temp_id])
                    shutil.rmtree(
                        os.path.join(self.output_dir, f"id_{temp_id}"),
                        ignore_errors=True
                    )
            except asyncio.CancelledError:
                pass

        self.timeout_tasks[temp_id] = asyncio.create_task(watcher())

    @staticmethod
    def _safe(x):
        return x if x is not None else "Unknow"

    async def maybe_put_id_to_queue(self, person_meta: dict):
        """
        Bổ sung thông tin vào person_meta và đưa vào hàng đợi nếu có age, gender hoặc race.
        """
        if not person_meta:
            return

        # Bỏ qua nếu thiếu toàn bộ age, gender, race
        if not any(
            (v := person_meta.get(k)) is not None and not (isinstance(v, str) and v.lower() == "none")
            for k in ("age", "gender", "race")
        ):
            logger.info("Bỏ qua, thiếu toàn bộ age, gender, race")
            return


        # Lấy dữ liệu head_point_3d một cách an toàn
        hp3d = person_meta.get("head_point_3d")
        if hp3d is None:
            point3D = [0, 0, 0]
        else:
            # Nếu là numpy array hoặc list/tuple
            point3D = list(hp3d)

        payload = {
            "frame_id":    person_meta.get("frame_id"),
            "person_id":   person_meta.get("person_id"),
            "gender":      self._safe(person_meta.get("gender")),
            "race":        self._safe(person_meta.get("race")),
            "age":         self._safe(person_meta.get("age")),
            "height":      float(person_meta.get("est_height_m") or 0.0),
            "time_detect": person_meta.get("time_detect"),
            "camera_id":   person_meta.get("camera_id"),
            "point3D":     point3D,
        }
        logger.info("[SEND ID] Đã put vào id_queue: %s", payload)
        await self.id_queue.put(payload)


    async def assign_id(self, **kwargs) -> str:
        """Main entry: match existing or create new ID."""
        logger.debug(f"[assign_id] Keys: {list(kwargs.keys())}")

        normalized = {
            "gender":          kwargs.get("gender"),
            "race":            kwargs.get("race"),
            "age":             kwargs.get("age"),
            "body_color":      kwargs.get("body_color"),
            "feature":         kwargs.get("feature_person"),
            "face_embedding":  kwargs.get("face_embedding"),
            "est_height_m":    kwargs.get("est_height_m"),
            "head_point_3d":   kwargs.get("head_point_3d"),
            "bbox_data":       kwargs.get("bbox"),
            "frame_id":        kwargs.get("frame_id"),
            "time_detect":     kwargs.get("time_detect"),
        }

        # drop time_detect for matching
        match_args = {k: v for k, v in normalized.items() if k != "time_detect"}
        matched = await self.reid_manager.match_id(**match_args)
        if matched:
            logger.info(f"[assign_id] Existing ID: {matched}")
            if matched not in self.temp_id:
                await self.handle_temp_update(matched, normalized, kwargs.get("keypoints"))
            else:
                # 1) update DB
                data = {
                    "gender": normalized["gender"],
                    "race": normalized["gender"],
                    "age": normalized["age"],
                }
                logger.info(f"[UPDATE ID] data: {data}")
                await self.reid_manager.update_person(matched, **normalized)

                # 2) fetch meta, enrich and enqueue
                person_meta = await self.reid_manager.get_person_meta(matched)
                if person_meta:
                    # merge in extra fields without mutating original
                    enriched = {
                        **person_meta,
                        "frame_id":      normalized["frame_id"],
                        "time_detect":   normalized["time_detect"],
                        "camera_id":     self.config.get("camera_id"),
                        **{k: normalized[k] for k in ("est_height_m", "head_point_3d")}
                    }
                    await self.maybe_put_id_to_queue(enriched)

            return matched

        # No match → create new
        logger.info("[assign_id] Creating new temp ID")
        temp_id = await self.reid_manager.create_person(**normalized)
        self.temp_id.add(temp_id)
        logger.debug(f"New temp ID: {temp_id}")
        await self._schedule_timeout(temp_id)
        return temp_id
    
    async def close(self):
        await self.rabbit.stop()
        await self.reid_manager.close()


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
