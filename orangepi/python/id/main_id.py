import os
import cv2
import numpy as np
import time
import asyncio
from collections import Counter
import torch
import uuid
import shutil
import subprocess
from collections import Counter, defaultdict

from feature.feature_osnet import FeatureModel 
from .reid_manager_new import ReIDManager
from face_processing.face_v2.face_analyze_new import FaceAnalyze
from rabbitMQ.rabbitMQ_manager import RabbitMQManager


from utils.logging_python_orangepi import get_logger 
logger = get_logger(__name__)

class PersonReID:
    def __init__( 
        self,
        id_socket_queue: asyncio.Queue,        
        output_dir="output_frames_id",
        feature_threshold=0.7,
        color_threshold=0.5,
        avg_threshold=0.8,
        top_k=3,
        thigh_weight=4, 
        torso_weight=4,
        feature_weight: float = 0.5, 
        color_weight: float = 0.5,
        db_path="database.db",
        device_id=None,
        temp_timeout=10,  # thời gian timeout (giây) cho mỗi temp_id
        min_detections=3,
        merge_threshold=0.75,
        face_threshold = 0.8,
    ):
        """Initialize PersonReID với Pub/Sub, TempDB, và timeout cho temp_id."""
        logger.info("Initializing PersonReID system...")
        self.OUTPUT_DIR = output_dir
        if os.path.exists(self.OUTPUT_DIR):
            logger.info(f"Xóa thư mục output cũ: {self.OUTPUT_DIR}")
            shutil.rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # RabbitMQ manager (remote lookup)
        self.device_id = device_id or str(uuid.uuid4())
        self.rabbit = RabbitMQManager(device_id=self.device_id)
        self.rabbit.on_request = self._handle_remote_request
        self._rabbit_task = asyncio.create_task(self.rabbit.start())

        # model
        self.feature_extractor = FeatureModel()
        self.reid_manager = ReIDManager(
            id_socket_queue=id_socket_queue,
            db_path=db_path,
            feature_threshold=feature_threshold,
            color_threshold=color_threshold,
            avg_threshold=avg_threshold,
            top_k=top_k,
            thigh_weight=thigh_weight, 
            torso_weight=torso_weight,
            feature_weight=feature_weight,
            color_weight=color_weight
        )
        self.face_analyze = FaceAnalyze() 

        # Thống kê
        self.frame_index = 0
        self.total_id_statistics = Counter()
        self.update_interval = 15
        self.last_check_time = time.time()
        self.fps_avg = 0.0
        self.call_count = 0
        self.db_path = db_path
        self.statistics_lock = asyncio.Lock()
        self.merge_threshold = merge_threshold

        # --- Dùng để quản lý temp_id ---
        self.temp_update_counts = defaultdict(int)  # đếm số lần gọi update_person cho mỗi temp_id
        self.remote_requested = set()  # đã từng gởi remote lookup hay chưa
        self._temp_timeout = temp_timeout  # thời gian chờ tối đa (giây) trước khi xóa temp_id
        self._timeout_tasks = {}  # lưu { temp_id: asyncio.Task } để hủy/reschedule khi cần
        self.min_detections = min_detections

        self.index_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()

        logger.info(
            f"PersonReID initialized (feat_th={feature_threshold}, color_th={color_threshold}, "
            f"device_id={self.device_id}, temp_timeout={self._temp_timeout}s)"
        )

    async def extract_feature_async(self, human_box):
        """Asynchronously extract feature."""
        start_time = time.time()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.feature_extractor.extract_feature, human_box)
        logger.debug(f"Feature extraction took {time.time() - start_time:.2f} seconds")
        return result

    async def face_processing(self, head_box):
        """Asynchronously process face data."""
        start_time = time.time()
        result = await self.face_analyze.analyze(head_box)
        return result

    async def save_crop_async(self, crop_path, human_box):
        """Asynchronously save cropped image as JPEG."""
        start_time = time.time()
        await asyncio.to_thread(cv2.imwrite, crop_path, human_box, [cv2.IMWRITE_JPEG_QUALITY, 100])
        logger.debug(f"Saving crop took {time.time() - start_time:.2f} seconds")

    async def _handle_remote_request(self, data: dict) -> str | None:
        """
        Callback invoked by RabbitMQManager when a remote lookup arrives.
        Compute match locally and return matched_id or None.
        """
        # extract payload
        gender = data.get('gender')
        race = data.get('race')
        age = data.get('age')
        body_color = np.array(data['body_color'], dtype=np.float32) if data.get('body_color') else None
        feature = np.array(data['feature'], dtype=np.float32) if data.get('feature') else None
        face_embedding = np.array(data['face_embedding'], dtype=np.float32) if data.get('face_embedding') else None
        # logger.warning("Đã nhận được 1 yêu cầu")
        # local match
        matched_id = await self.reid_manager.match_id(
            new_gender=gender,
            new_race=race,
            new_age=age,
            new_body_color=body_color,
            new_feature=feature,
            new_face_embedding=face_embedding,
        )
        return matched_id


    async def _nearest_feature_matching(self, temp_id: str):
        """
        Check nearest existing person by ID and replace if similarity >= threshold.
        Returns True if replaced, False otherwise.
        """
        closest_id, sim_score = await self.reid_manager.find_closest_by_id(temp_id)
        threshold = self.merge_threshold
        if closest_id and sim_score >= threshold:
            logger.warning(f"MERGE: {temp_id} → {closest_id} (sim={sim_score:.3f})")
            await self.reid_manager.merge_temp_person_id(temp_id, closest_id)

            # Cleanup
            self.temp_update_counts.pop(temp_id, None)

            # Cancel timeout task
            if temp_id in self._timeout_tasks:
                self._timeout_tasks.pop(temp_id).cancel()

            # Move folder using shell mv
            src_folder = os.path.join(self.OUTPUT_DIR, f"id_{temp_id}")
            dst_folder = os.path.join(self.OUTPUT_DIR, f"id_{closest_id}", f"id_{temp_id}")
            if os.path.exists(src_folder):
                os.makedirs(os.path.dirname(dst_folder), exist_ok=True)
                try:
                    subprocess.run(["mv", src_folder, dst_folder], check=True)
                    shutil.rmtree(src_folder)
                    logger.info(f"Moved folder {src_folder} into {os.path.dirname(dst_folder)}")
                except subprocess.CalledProcessError as e:
                    logger.exception(f"Failed to move folder {src_folder} to {dst_folder}: {e}")

            return True

        logger.warning(f"NO MERGE: {temp_id} → {closest_id} (sim={sim_score:.3f})")
        return False


    async def _publish_and_replace_temp_id(
        self,
        temp_id: str,
        gender: str | None,
        race: str | None,
        age: str | None,
        body_color: np.ndarray | list | None,
        feature_person: np.ndarray | list,
        face_embedding: np.ndarray | list | None,
    ):
        """
        Publish remote lookup for temp_id after min_detections and handle response.
        """
        try:
            await asyncio.wait_for(self.rabbit._ready.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            # broker chưa kết nối: bỏ qua remote lookup
            logger.debug(f"[{temp_id}] RabbitMQ not ready, skipping lookup")
            return

        if not self.rabbit._connected:
            logger.warning(f"[{temp_id}] RabbitMQ not connected—skip remote lookup")
            return
        
        self.remote_requested.add(temp_id)
        payload = {
            "gender": gender,
            "race": race,
            "age": age,
            "body_color": (
                np.asarray(body_color, dtype=float).tolist()
                if body_color is not None else None
            ),
            "feature": np.asarray(feature_person, dtype=float).tolist(),
            "body_color": (
                np.asarray(body_color, dtype=float).tolist()
                if body_color is not None else None
            ),
            "face_embedding": (
                np.asarray(face_embedding, dtype=float).tolist()
                if face_embedding is not None else None
            ),
        }
        fut = await self.rabbit.publish_remote_request(payload, request_id=str(temp_id))
        logger.info(f"► Published remote lookup cho temp_id: {temp_id}")

        async def handle_response():
            try:
                matched = await asyncio.wait_for(fut, timeout=20) 
                if matched and matched != temp_id:
                    logger.warning(f"Remote match: {temp_id} → {matched}")
                    await self.reid_manager.replace_temp_person_id(temp_id, matched)
                else:
                    logger.warning(f"No better match found remotely for {temp_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Remote lookup timeout for temp_id: {temp_id}")

        asyncio.create_task(handle_response())

    async def _handle_temp_id_update(
        self,
        temp_id: str,
        gender: str | None,
        race: str | None,
        age: str | None,
        body_color: np.ndarray | list | None,
        feature_person: np.ndarray | list,
        face_embedding: np.ndarray | list | None,
        map_keypoints: list[tuple[float, float]],
    ):
        """
        Update temp_id counter, reset timeout, try nearest-feature matching;
        if below threshold and has enough valid keypoints, publish remote lookup,
        else reschedule timeout.
        """
        # 1) Increment count
        self.temp_update_counts[temp_id] += 1
        count = self.temp_update_counts[temp_id]
        logger.debug(f"Temp ID {temp_id} update count: {count}")

        # 2) Cancel previous watcher to reset timeout
        if temp_id in self._timeout_tasks:
            self._timeout_tasks.pop(temp_id).cancel()

        # 3) Try nearest feature matching
        replaced = await self._nearest_feature_matching(temp_id)
        if replaced:
            return

        # 4) If has enough valid keypoints, publish remote and replace
        if has_enough_valid_keypoints(map_keypoints):
            await self._publish_and_replace_temp_id(
                temp_id, gender, race, age, body_color, feature_person, face_embedding
            )
        else:
            # Reschedule timeout watcher
            await self.start_temp_timeout_watcher(temp_id)

    async def start_temp_timeout_watcher(self, temp_id: str):
        """
        Tạo hoặc reschedule một task để theo dõi timeout cho temp_id.
        Nếu sau self._temp_timeout giây mà temp_id chưa đủ update (temp_update_counts < 3)
        và chưa gởi remote lookup, thì xóa temp_id.
        """

        # Nếu watcher cũ đang chạy → hủy để reschedule lại
        if temp_id in self._timeout_tasks:
            self._timeout_tasks[temp_id].cancel()

        async def _watch():
            try:
                await asyncio.sleep(self._temp_timeout)

                # Sau khi ngủ đủ temp_timeout
                count = self.temp_update_counts.get(temp_id, 0)
                if temp_id not in self.remote_requested and count < 3:
                    logger.warning(
                        f"[TIMEOUT] temp_id {temp_id} chỉ update {count} lần (<3) → xóa."
                    )
                    await self.reid_manager.remove_low_person_ids([temp_id])

                    # Dọn dẹp thống kê
                    async with self.statistics_lock:
                        self.total_id_statistics.pop(temp_id, None)

                    # Xóa thư mục output nếu có
                    folder = os.path.join(self.OUTPUT_DIR, f"id_{temp_id}")
                    if os.path.exists(folder):
                        shutil.rmtree(folder)
                        logger.info(f"Removed directory: {folder}")

                # Sau khi xử lý xong, remove task tracking
                self._timeout_tasks.pop(temp_id, None)

            except asyncio.CancelledError:
                # Bị hủy để reschedule (có update mới)
                logger.debug(f"Timeout watcher bị hủy cho temp_id: {temp_id}")
            except Exception as e:
                logger.error(f"Lỗi trong watcher cho {temp_id}: {e}")
                self._timeout_tasks.pop(temp_id, None)

        task = asyncio.create_task(_watch())
        self._timeout_tasks[temp_id] = task

    async def process_id(
        self,
        gender_result,
        race_result,
        age_result,
        body_color,
        feature_person,
        face_embedding,
        est_height_m,
        world_point_3d,
        bbox,
        frame_id,
        map_keypoints,
        time_detect,
    ) -> str:
        """
        Gán person_id. Truyền các thuộc tính vật lý (chiều cao, tọa độ 3D) vào reid_manager.
        """
        gender = gender_result
        race = race_result
        age = age_result

        # ### <<< SỬA LỖI Ở ĐÂY
        # Tạo một chuỗi an toàn để log, xử lý trường hợp est_height_m là None
        height_str = f"{est_height_m:.2f}m" if est_height_m is not None else "N/A"
        logger.debug(f"Begin process_id: gender={gender}, race={race}, age={age}, height={height_str}")
        # ### <<< KẾT THÚC SỬA LỖI

        # 1. Thử match local với đầy đủ thông tin
        local_id = await self.reid_manager.match_id(
            gender,
            race,
            age,
            body_color,
            feature_person,
            face_embedding,
            est_height_m,     # Thêm chiều cao
            world_point_3d,    # Thêm tọa độ 3D
            bbox,
            frame_id,
        )

        if local_id:
            logger.info(f"Matched local person_id: {local_id}")

            if local_id not in self.remote_requested:
                # Nếu là temp_id, xử lý song song việc update và kiểm tra gửi remote
                # --- Đã sửa: thiếu body_color trong _handle_temp_id_update ---
                await asyncio.gather(
                    self._handle_temp_id_update(
                        temp_id=local_id,
                        gender=gender,
                        race=race,
                        age=age,
                        body_color=body_color, # Bổ sung tham số này
                        feature_person=feature_person,
                        face_embedding=face_embedding,
                        map_keypoints=map_keypoints,
                    ),
                    self.reid_manager.update_person(
                        local_id,
                        gender,
                        race,
                        age,
                        body_color, # Thêm body_color
                        feature_person,
                        face_embedding,
                        est_height_m,
                        world_point_3d,
                        bbox,
                        frame_id,
                        time_detect
                    )
                )
            else:
                # Với ID đã xác định, chỉ cần update DB
                await self.reid_manager.update_person(
                    local_id,
                    gender,
                    race,
                    age,
                    body_color,
                    feature_person,
                    face_embedding,
                    est_height_m,
                    world_point_3d,
                    bbox,
                    frame_id,
                    time_detect
                )
            return local_id

        # 2. No match → tạo mới temp_id với đầy đủ thông tin
        temp_id = await self.reid_manager.create_person(
            gender,
            race,
            age,
            body_color,
            feature_person,
            face_embedding,
            est_height_m,
            world_point_3d,
            bbox,
            frame_id,
            time_detect
        )
        logger.info(f"No local match → created temp_id: {temp_id}")

        # Khởi tạo counter và watcher cho temp_id mới
        self.temp_update_counts[temp_id] = 0
        await self.start_temp_timeout_watcher(temp_id)

        return temp_id

                # 2. Xử lý tọa độ thế giới (thêm Z=0 để debug)
        world_point_xy = data.get('world_point_xy')
        world_point_xyz = (*world_point_xy, 0.0) if world_point_xy is not None else None

    async def analyze(self, data: dict, max_retries=3):
        """Xử lý một frame với phát hiện người và ReID một cách bất đồng bộ."""
        if not isinstance(data, dict):
            logger.error(f"data must be a dict, got {type(data)}")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                            # 2. Xử lý tọa độ thế giới (thêm Z=0 để debug)
                world_point_xy = data.get('world_point_xy')
                world_point_xyz = (*world_point_xy, 0.0) if world_point_xy is not None else None
                # Lấy dữ liệu khớp với output của process_frame_queue
                frame_id = data.get("frame_id")
                human_image = data.get("human_box")
                body_parts = data.get("body_parts", {})
                body_color = data.get("body_color")
                bbox = data.get("bbox")
                map_keypoints = data.get("map_keypoints")
                
                # Lấy các thông tin bổ sung
                camera_id = data.get("camera_id", "unknown_cam")
                distance_mm = data.get("distance_mm")
                est_height_m = data.get("est_height_m")
                world_point_3d = world_point_xyz

                time_detect = data.get("time_detect")

                if human_image is None or human_image.size == 0:
                    logger.error(f"Missing or empty human_box in data for frame_id {frame_id}")
                    return None

                # ... (Phần xử lý feature và face không đổi) ...
                if hasattr(self, "extract_feature_async"):
                    if asyncio.iscoroutinefunction(self.extract_feature_async):    
                        task_feat = self.extract_feature_async(human_image)
                    else:
                        task_feat = asyncio.to_thread(self.extract_feature_async, human_image)
                else:
                    task_feat = asyncio.to_thread(self.extract_feature, human_image)
                tasks = [task_feat]

                head_bbox = body_parts.get("head")
                if head_bbox:
                    x1, y1, x2, y2 = head_bbox
                    if x2 > x1 and y2 > y1:
                        head_crop = human_image[y1:y2, x1:x2]
                        if getattr(head_crop, "size", 0) > 0:
                            if hasattr(self, "face_processing"):
                                logger.info("Đã đi qua face processing")
                                if asyncio.iscoroutinefunction(self.face_processing):
                                    tasks.append(self.face_processing(head_crop))
                                else:
                                    tasks.append(asyncio.to_thread(self.face_processing, head_crop))
                            else:
                                logger.warning("Không có method face_processing, bỏ qua head.")
                        else:
                            logger.warning(f"Empty head_crop at {head_bbox}")
                    else:
                        logger.warning(f"Invalid head_bbox: {head_bbox}")
                else:
                    logger.warning("Khong co head_bbox.")

                results = await asyncio.gather(*tasks, return_exceptions=True)
                feature_person = results[0]
                if isinstance(feature_person, Exception):
                    raise feature_person

                face_results = results[1] if len(results) > 1 else None
                if face_results:
                    try:
                        face_embedding, age_result, gender_result, emotion_result, race_result = face_results
                    except Exception:
                        face_embedding = age_result = gender_result = emotion_result = race_result = None
                else:
                    face_embedding = age_result = gender_result = emotion_result = race_result = None
                
                # ### <<< THAY ĐỔI: Cập nhật lời gọi hàm process_id
                if asyncio.iscoroutinefunction(self.process_id):
                    final_id = await self.process_id(
                        gender_result,
                        race_result,
                        age_result,
                        body_color,
                        feature_person,
                        face_embedding,
                        est_height_m,    # Thêm chiều cao
                        world_point_3d,   # Thêm tọa độ 3D
                        bbox,
                        frame_id,
                        map_keypoints,
                        time_detect
                    )
                else:
                    final_id = await asyncio.to_thread(
                        self.process_id,
                        gender_result,
                        race_result,
                        age_result,
                        body_color,
                        feature_person,
                        face_embedding,
                        est_height_m,    # Thêm chiều cao
                        world_point_3d,   # Thêm tọa độ 3D
                        bbox,
                        frame_id,
                        map_keypoints,
                        time_detect
                    )
                # ### <<< KẾT THÚC THAY ĐỔI

                # ... (Phần còn lại của hàm không đổi) ...
                id_folder = os.path.join(self.OUTPUT_DIR, f"id_{final_id}")
                await asyncio.to_thread(os.makedirs, id_folder, exist_ok=True)
                async with self.index_lock:
                    current_index = self.frame_index
                    self.frame_index += 1
                crop_filename = f"frame_{current_index}.jpg"
                crop_path = os.path.join(id_folder, crop_filename)
                if hasattr(self, "save_crop_async") and asyncio.iscoroutinefunction(self.save_crop_async):
                    await self.save_crop_async(crop_path, human_image)
                else:
                    await asyncio.to_thread(self.save_crop, crop_path, human_image)
                async with self.statistics_lock:
                    self.total_id_statistics.setdefault(final_id, 0)
                    self.total_id_statistics[final_id] += 1
                current_time = time.time()
                if current_time - self.last_check_time >= self.update_interval:
                    logger.info("Đã qua update_interval, có thể xử lý thống kê định kỳ ở đây")
                    self.last_check_time = current_time
                duration = current_time - start_time
                fps_current = 1 / duration if duration > 0 else 0.0
                async with self.stats_lock:
                    self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
                logger.info(
                    f"Processed frame {current_index} for ID_{final_id} from Cam_{camera_id}. "
                    # f"Dist: {distance_mm/1000:.2f}m, Height: {est_height_m:.2f}m. FPS_ID = {self.fps_avg:.2f}"
                )
                return final_id

            except asyncio.CancelledError:
                logger.info("analyze bị hủy bởi CancelledError")
                raise
            except Exception as e:
                logger.error(f"Error processing frame (attempt {attempt}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts.")
                    return None
                
    async def close(self):
        """Cleanup connections and resources."""
        await self.rabbit.stop()
        await self.reid_manager.close()

def has_enough_valid_keypoints(keypoints, min_valid_keypoints=14):
    valid_count = sum(1 for x, y in keypoints if x != 0 and y != 0)
    return valid_count >= min_valid_keypoints

async def start_id(processing_queue: asyncio.Queue, person_reid: PersonReID = None):
    """Asynchronous function to process frames từ processing_queue."""
    logger.info("Starting ID processing")
    created_local = False
    if person_reid is None:
        from config import DEFAULT_CONFIG
        allowed = [
            "output_dir",
            "feature_threshold",
            "color_threshold",
            "db_path",
            "rabbitmq_url",
            "cleanup_interval",
            "min_detections",
        ]
        filtered = {k: v for k, v in DEFAULT_CONFIG.items() if k in allowed}
        person_reid = PersonReID(**filtered)
        created_local = True

    try:
        while True:
            try:
                data = await processing_queue.get()
            except asyncio.CancelledError:
                logger.info("start_id nhận CancelledError khi get queue")
                raise

            if data is None:
                processing_queue.task_done()
                logger.info("Received sentinel None, kết thúc start_id")
                break

            try:
                final_id = await person_reid.analyze(data)
                # Xử lý final_id nếu cần
            except asyncio.CancelledError:
                logger.info("start_id: analyze bị hủy")
                raise
            except Exception as e:
                logger.error(f"Error in analyze: {e}", exc_info=True)
            finally:
                processing_queue.task_done()
    finally:
        logger.info("ID processing loop đã dừng.")
        # Đảm bảo đóng person_reid (bao gồm RabbitMQ và ReID manager) nếu được tạo cục bộ
        if created_local:
            logger.info("Closing PersonReID and associated managers...")
            try:
                await person_reid.close()
            except Exception as close_e:
                logger.error(f"Lỗi khi đóng person_reid: {close_e}", exc_info=True)

if __name__ == "__main__":
    async def main():
        frame_queue = asyncio.Queue()
        await start_id(frame_queue)
    asyncio.run(main()) 