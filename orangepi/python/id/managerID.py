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


def has_enough_valid_keypoints(keypoints, min_valid_keypoints=14):
    if not keypoints or not isinstance(keypoints, (list, tuple)):
        return False
    return sum(1 for x, y in keypoints if x != 0 and y != 0) >= min_valid_keypoints


class ManagerID:
    """Handles ReID operations, DB, feature and face extraction, and remote lookups."""

    def __init__(self, id_queue: asyncio.Queue, output_dir: str, config: dict):
        self.config = config
        self.id_queue = id_queue
        self.output_dir = output_dir
        self.device_id = self.config.get('device_id', str(uuid.uuid4()))

        # RabbitMQ setup
        self.rabbit = RabbitMQManager(device_id=self.device_id)
        self.rabbit.on_request = self._handle_remote_request
        self._rabbit_task = asyncio.create_task(self.rabbit.start())

        # Feature and face models
        self.feature_extractor = FeatureModel()
        self.face_analyze = FaceAnalyze()

        # ReID manager
        self.reid_manager = ReIDManager(
            db_path=self.config['db_path'],
            feature_threshold=self.config['feature_threshold'],
            face_threshold=self.config['face_threshold'],
            hard_feature_threshold=self.config['hard_feature_threshold'],
            hard_face_threshold=self.config['hard_face_threshold'],
            top_k=self.config['top_k'],
        )

        self.merge_threshold = self.config['merge_threshold']
        self.temp_timeout = self.config['temp_timeout']
        self.min_detections = self.config['min_detections']

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
        """Try to find a closest match and merge."""
        try:
            closest_id, score = await self.reid_manager.find_closest_by_id(temp_id)
            if closest_id and score >= self.merge_threshold:
                await self.reid_manager.merge_temp_person_id(temp_id, closest_id)
                self._move_folder(temp_id, closest_id)
                return closest_id
        except Exception:
            logger.exception(f"Error merging temp {temp_id}")
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
        """Update or promote a temp ID based on detections and merge/remote logic."""
        # 1) Count this detection
        self.temp_counts[person_id] += 1

        # 2) Local merge if possible
        merged = await self._find_and_merge(person_id)
        if merged:
            # cancel and remove timeout task
            task = self.timeout_tasks.pop(person_id, None)
            if task:
                task.cancel()
            # discard temp
            self.temp_id.discard(person_id)
            # update and enqueue merged ID
            await self.reid_manager.update_person(merged, **normalized)
            meta = await self.reid_manager.get_person_meta(merged)
            if meta:
                enriched = {
                    **meta,
                    'frame_id': normalized.get('frame_id'),
                    'time_detect': normalized.get('time_detect'),
                    'camera_id': self.config.get('camera_id'),
                    'est_height_m': normalized.get('est_height_m'),
                    'head_point_3d': normalized.get('head_point_3d')
                }
                await self.maybe_put_id_to_queue(enriched)
            return

        # 3) Promote to official when reach threshold
        if self.temp_counts[person_id] >= self.min_detections:
            logger.info(f"[PROMOTE] Temp ID {person_id} is now official after {self.temp_counts[person_id]} detections")
            # Cancel and remove timeout task
            task = self.timeout_tasks.pop(person_id, None)
            if task:
                task.cancel()
            # discard temp and count
            self.temp_id.discard(person_id)
            self.temp_counts.pop(person_id, None)
            # enqueue official ID
            await self.maybe_put_id_to_queue({
                **normalized,
                'person_id': person_id,
                'frame_id': normalized.get('frame_id'),
                'time_detect': normalized.get('time_detect'),
                'camera_id': self.config.get('camera_id'),
                'est_height_m': normalized.get('est_height_m'),
                'head_point_3d': normalized.get('head_point_3d')
            })
            return

        # 4) Not merged or promoted: update temp in DB
        await self.reid_manager.update_person(person_id, **normalized)
        # ensure person_id marked as temp
        self.temp_id.add(person_id)

        # 5) Decide remote request or schedule timeout
        if has_enough_valid_keypoints(keypoints):
            await self._publish_remote(
                person_id,
                normalized.get('gender'), normalized.get('race'), normalized.get('age'),
                normalized.get('body_color'), normalized.get('feature'), normalized.get('face_embedding')
            )
        else:
            await self._schedule_timeout(person_id)

    async def _publish_remote(self, temp_id: str, gender, race, age, body_color, feature, face_embedding):
        """Send to remote ReID and process response."""
        try:
            await asyncio.wait_for(self.rabbit._ready.wait(), timeout=1)
            self.remote_requested.add(temp_id)
            payload = dict(
                gender=gender, race=race, age=age,
                body_color=body_color, feature=feature,
                face_embedding=face_embedding
            )
            fut = await self.rabbit.publish_remote_request(payload, request_id=str(temp_id))

            async def on_response():
                try:
                    matched = await asyncio.wait_for(fut, timeout=20)
                    if matched and matched != temp_id:
                        # cancel and remove timeout task
                        task = self.timeout_tasks.pop(temp_id, None)
                        if task:
                            task.cancel()
                        # replace and discard temp
                        await self.reid_manager.replace_temp_person_id(temp_id, matched)
                        self.temp_id.discard(temp_id)
                        # enqueue merged as official
                        meta = await self.reid_manager.get_person_meta(matched)
                        if meta:
                            enriched = {
                                **meta,
                                'frame_id': None,
                                'time_detect': None,
                                'camera_id': self.config.get('camera_id'),
                                'est_height_m': None,
                                'head_point_3d': None
                            }
                            await self.maybe_put_id_to_queue(enriched)
                except asyncio.TimeoutError:
                    logger.warning(f"Remote timeout for {temp_id}")
                    # fallback: clean flag and reschedule timeout
                    self.remote_requested.discard(temp_id)
                    if temp_id not in self.timeout_tasks:
                        await self._schedule_timeout(temp_id)
                except Exception:
                    logger.exception("Error handling remote response")
                    self.remote_requested.discard(temp_id)
                    if temp_id not in self.timeout_tasks:
                        await self._schedule_timeout(temp_id)
                finally:
                    # ensure flag cleaned
                    self.remote_requested.discard(temp_id)

            asyncio.create_task(on_response())
        except Exception:
            logger.exception(f"Error publishing remote for {temp_id}")
            self.remote_requested.discard(temp_id)
            await self._schedule_timeout(temp_id)

    async def _schedule_timeout(self, temp_id: str):
        """Schedule removal of low-count temps after timeout."""
        # Cancel & remove old task if exists
        old = self.timeout_tasks.pop(temp_id, None)
        if old:
            old.cancel()

        async def watcher():
            try:
                await asyncio.sleep(self.temp_timeout)
                # if still temp and not requested remotely and below threshold
                if (
                    temp_id in self.temp_id and
                    temp_id not in self.remote_requested and
                    self.temp_counts.get(temp_id, 0) < self.min_detections
                ):
                    # clean state
                    self.temp_id.discard(temp_id)
                    self.remote_requested.discard(temp_id)
                    self.temp_counts.pop(temp_id, None)
                    # remove from DB and filesystem
                    await self.reid_manager.remove_low_person_ids([temp_id])
                    shutil.rmtree(os.path.join(self.output_dir, f"id_{temp_id}"), ignore_errors=True)
                    self.timeout_tasks.pop(temp_id, None)
            except asyncio.CancelledError:
                pass

        self.timeout_tasks[temp_id] = asyncio.create_task(watcher())

    @staticmethod
    def _safe(x):
        return x if x is not None else "Unknown"

    async def maybe_put_id_to_queue(self, person_meta: dict):
        """Enqueue person_meta if valid."""
        if not person_meta:
            return
        if not any(
            (v := person_meta.get(k)) is not None and not (isinstance(v, str) and v.lower() == "none")
            for k in ("age", "gender", "race")
        ):
            logger.info("Skip enqueue: missing all age/gender/race")
            return

        hp3d = person_meta.get("head_point_3d")
        point3D = list(hp3d) if hp3d is not None else [0, 0, 0]

        payload = {
            "frame_id": person_meta.get("frame_id"),
            "person_id": person_meta.get("person_id"),
            "gender": self._safe(person_meta.get("gender")),
            "race": self._safe(person_meta.get("race")),
            "age": self._safe(person_meta.get("age")),
            "height": float(person_meta.get("est_height_m") or 0.0),
            "time_detect": person_meta.get("time_detect"),
            "camera_id": person_meta.get("camera_id"),
            "point3D": point3D,
        }
        logger.info("[SEND ID] Enqueue: %s", payload)
        await self.id_queue.put(payload)

    async def assign_id(self, **kwargs) -> str:
        """Main entry: match or create ID."""
        logger.debug(f"[assign_id] Keys: {list(kwargs.keys())}")

        normalized = {k: kwargs.get(k) for k in (
            "gender", "race", "age", "body_color", "feature_person",
            "face_embedding", "est_height_m", "head_point_3d",
            "bbox", "frame_id", "time_detect"
        )}
        normalized['feature'] = normalized.pop('feature_person')
        normalized['bbox_data'] = normalized.pop('bbox')

        match_args = {k: v for k, v in normalized.items() if k != "time_detect"}
        matched = await self.reid_manager.match_id(**match_args)
        if matched:
            if matched in self.temp_id:
                await self.handle_temp_update(matched, normalized, kwargs.get("keypoints"))
            else:
                await self.reid_manager.update_person(matched, **normalized)
                meta = await self.reid_manager.get_person_meta(matched)
                if meta:
                    enriched = {**meta,
                                "frame_id": normalized["frame_id"],
                                "time_detect": normalized["time_detect"],
                                "camera_id": self.config.get("camera_id"),
                                "est_height_m": normalized.get("est_height_m"),
                                "head_point_3d": normalized.get("head_point_3d")}
                    await self.maybe_put_id_to_queue(enriched)
            return matched

        # No match → create new temp
        temp_id = await self.reid_manager.create_person(**normalized)
        logger.info(f"[CREATE ID] Đã tạo id mới {temp_id}")
        self.temp_id.add(temp_id)
        self.temp_counts[temp_id] = 1  # NEW: Count the first detection
        await self._schedule_timeout(temp_id)
        return temp_id
