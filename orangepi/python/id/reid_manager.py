import os
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

import numpy as np
import aiosqlite
import uuid
import sqlite_vec
from sklearn.metrics.pairwise import cosine_similarity
from utils.pose_color_signature_new import preprocess_color
from track_local.track import TrackingManager

from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class ReIDManager:
    def __init__(
        self,
        db_path='database.db',
        feature_threshold=0.75,
        color_threshold=0.6,
        avg_threshold=0.8,
        top_k=3,
        global_color_mean=None,
        thigh_weight: float = 4,
        torso_weight: float = 4, 
        feature_weight: float = 0.5,
        color_weight: float = 0.5,
        δ0_feat = 0.25,
        δ_margin = 0.05,
        face_threshold = 0.8,
    ):
        self.db_path = db_path
        self.feature_threshold = feature_threshold
        self.color_threshold = color_threshold 
        self.avg_threshold = avg_threshold
        self.top_k = top_k
        self.face_threshold = face_threshold


        # incremental stats for global mean
        self.sum_color = np.zeros(51, dtype=np.float64)
        self.count_color = np.zeros(51, dtype=np.int64)
        if global_color_mean is not None:
            init = np.array(global_color_mean, dtype=np.float32)
            valid = ~np.isnan(init)
            self.sum_color[valid] = init[valid]
            self.count_color[valid] = 1
        self.global_color_mean = np.divide(
            self.sum_color,
            np.where(self.count_color > 0, self.count_color, 1),
            out=np.zeros_like(self.sum_color, dtype=np.float32),
            where=self.count_color > 0
        ).astype(np.float32)

        # ------------------------------------------------------------------
        # THÊM: trọng số cho thigh & torso trên 45-dim
        # ------------------------------------------------------------------
        self.thigh_weight = thigh_weight  
        self.torso_weight = torso_weight

        # history for moving average per person
        self.feature_history = defaultdict(lambda: deque(maxlen=20))
        self.color_history   = defaultdict(lambda: deque(maxlen=20))
        self.face_embedding_history = defaultdict(lambda: deque(maxlen=20))

        # initialization
        self.db_lock = asyncio.Lock()
        self.db_init = asyncio.Event()
        self.db = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vec_ext = '/usr/local/lib/vec0.so'
        self.connection = None
        self.channel = None
        self.alias_map: dict[str, str] = {} 

        # start init tasks
        asyncio.create_task(self._init_database())

        self.tracking_manager = TrackingManager(
            max_time_lost=20,           # ví dụ 30
            proximity_thresh=0.7,                # tuỳ chỉnh
            appearance_thresh=0.5,               # tuỳ chỉnh
            feature_history_len=5
        )

        # asyncio.create_task(self.setup_rabbitmq(rabbitmq_url))
        logger.info("ReIDManager init successful")

    async def _init_database(self):
        async with self.db_lock: 
            # # 1) Tạo schema (bằng script bên ngoài)
            # from database.create_db import initialize_db
            # self.use_vec = await initialize_db(self.db_path, self.vec_ext)

            # 2) Mở kết nối
            db = await aiosqlite.connect(self.db_path)
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.enable_load_extension(True)
            # await db.load_extension(self.vec_ext)
            await db.load_extension(sqlite_vec.loadable_path())

            # 3) Bản ghi mặc định cho Cameras & Frames
            #    (chỉ thêm nếu chưa có – tránh lỗi trùng khóa)
            await db.execute(
                "INSERT OR IGNORE INTO Cameras(camera_id, location, resolution, model) "
                "VALUES('edge', 'unknown', 'unknown', 'unknown');"
            )

            ts = asyncio.get_event_loop().time()
            await db.execute(
                "INSERT OR IGNORE INTO Frames(frame_id, timestamp, camera_id) "
                "VALUES('frame0', ?, 'edge');",
                (ts,)
            )

            await db.commit()

            # 4) Hoàn tất
            self.db = db
            self.db_init.set()
            logger.info("Database ready, schema initialized, and default rows inserted.")


    async def _ensure_db_ready(self):
        await self.db_init.wait()

    def normalize(self, v: np.ndarray) -> np.ndarray:
        v = v.flatten()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


    def preprocess_color(self, body_color: np.ndarray):
        # flat = body_color.flatten().astype(np.float32)
        # mask = (~np.isnan(flat)).astype(np.float32)
        # filled = np.where(np.isnan(flat), self.global_color_mean, flat)
        # normed = self.normalize(filled)
        # return normed, mask
        weighted, mask = preprocess_color(body_color, self.thigh_weight, self.torso_weight)
        normed = self.normalize(weighted)
        return normed, mask
    
    def _update_global_color_stats(self, color_mean: np.ndarray):
        valid = ~np.isnan(color_mean)
        self.sum_color[valid] += color_mean[valid]
        self.count_color[valid] += 1
        self.global_color_mean[valid] = (
            self.sum_color[valid] / self.count_color[valid]
        ).astype(np.float32)


    async def create_person(self, gender, race, age, body_color, feature, face_embedding,
                            bbox_data=None, frame_id=None):
        """
        Thêm optional bbox_data (list/ndarray [x1,y1,x2,y2]) và frame_id (int) để tạo track ngay.
        """
        await self._ensure_db_ready()   
        try:
            # 1) Chuẩn bị embedding/color/face
            feat_v = self.normalize(np.array(feature, dtype=np.float32)) if feature is not None else None
            color_v, _ = self.preprocess_color(np.array(body_color, dtype=np.float32)) if body_color is not None else (None, None)
            face_v = self.normalize(np.array(face_embedding, dtype=np.float32)) if face_embedding is not None else None

            pid = str(uuid.uuid4())

            async with self.db_lock:
                # 2) Tạo bản ghi Persons
                await self.db.execute("INSERT INTO Persons(person_id) VALUES (?);", (pid,))

                # 3) Chèn embedding vào PersonsVec
                await self.db.execute(
                    "INSERT INTO PersonsVec(person_id, feature_mean, body_color_mean) VALUES(?, vec_f32(?), vec_f32(?));",
                    (
                        pid,
                        feat_v.tobytes() if feat_v is not None else None,
                        color_v.tobytes() if color_v is not None else None,
                    )
                )
                # 4) FaceVector nếu có
                if face_v is not None: 
                    await self.db.execute(
                        "INSERT OR REPLACE INTO FaceVector(person_id, face_embedding) VALUES(?, vec_f32(?))",
                        (pid, face_v.tobytes())
                    )
                # 5) Metadata
                await self.db.execute(
                    "INSERT INTO PersonsMeta(person_id, age, gender, race) VALUES (?, ?, ?, ?);",
                    (pid, age, gender, race)
                )
                # 6) Detection khởi tạo trong DB (không liên quan track manager, nhưng lưu log)
                ts = asyncio.get_event_loop().time()
                # Nếu muốn lưu frame_id thật, có thể truyền frame_id vào cột frame_id thay 'frame0'
                # db_frame = frame_id if frame_id is not None else 'frame0'
                await self.db.execute(
                    "INSERT INTO Detections(person_id, timestamp, camera_id, frame_id) VALUES(?, ?, ?, ?);",
                    (pid, ts, None, None)
                )

                await self.db.commit()

            # checkpoint WAL
            async with self.db_lock:
                await self.db.execute("PRAGMA wal_checkpoint(FULL);")
                await self.db.commit()

            # 7) Cập nhật history RAM
            if feat_v is not None:
                self.feature_history[pid].append(feat_v)
            if color_v is not None:
                self.color_history[pid].append(color_v)
                self._update_global_color_stats(color_v)
            if face_v is not None:
                self.face_embedding_history[pid].append(face_v)

            # 8) Thêm vào tracking_manager nếu có bbox_data và frame_id
            if bbox_data is not None and frame_id is not None:
                # feature đã normalized là feat_v
                # Gọi add_track
                self.tracking_manager.add_track(pid, bbox_data, frame_id, feature=feat_v)
                logger.info("Đã thêm ID vào tracking_manager")
            return pid

        except Exception as e:
            logger.exception(f"[DB] Failed to create person: {e}")
            return None


    async def update_person(self, person_id, gender=None, race=None, age=None,
                            body_color=None, feature=None, face_embedding=None,
                            bbox_data=None, frame_id=None):
        """
        Thêm optional bbox_data và frame_id để update track.
        Các tham số metadata nếu None sẽ không cập nhật.
        """
        await self._ensure_db_ready()
        try:
            # Remap nếu alias
            person_id = self.alias_map.get(person_id, person_id)

            # 1) Chuẩn bị embedding/color/face
            feat_v = self.normalize(np.array(feature, dtype=np.float32)) if feature is not None else None
            color_v, _ = self.preprocess_color(np.array(body_color, dtype=np.float32)) if body_color is not None else (None, None)
            face_v = self.normalize(np.array(face_embedding, dtype=np.float32)) if face_embedding is not None else None

            async with self.db_lock:
                ts = asyncio.get_event_loop().time()
                # 2) Ghi detection
                # db_frame = frame_id if frame_id is not None else 'frame0'
                await self.db.execute(
                    "INSERT INTO Detections(person_id, timestamp, camera_id, frame_id) VALUES (?, ?, ?, ?);",
                    (person_id, ts, None, None)
                )

                # 3) Cập nhật lịch sử RAM
                if feat_v is not None:
                    self.feature_history[person_id].append(feat_v)
                if color_v is not None:
                    self.color_history[person_id].append(color_v)
                    self._update_global_color_stats(color_v)
                if face_v is not None:
                    self.face_embedding_history[person_id].append(face_v)

                # 4) Tính moving average
                # mv_feat = (np.mean(self.feature_history[person_id], axis=0)
                #            if self.feature_history[person_id] else None)
                # mv_color = (np.mean(self.color_history[person_id], axis=0)
                #             if self.color_history[person_id] else None)
                # mv_face = (self.face_embedding_history[person_id][-1]
                #            if self.face_embedding_history[person_id] else None)
                # 4) Tính EMA song song
                mv_feat_coro  = asyncio.to_thread(compute_ema, self.feature_history[person_id])
                mv_color_coro = asyncio.to_thread(compute_ema, self.color_history[person_id])
                mv_face_coro  = asyncio.to_thread(compute_ema, self.face_embedding_history[person_id])

                mv_feat, mv_color, mv_face = await asyncio.gather(
                    mv_feat_coro,
                    mv_color_coro,
                    mv_face_coro,
                )

                # 5) UPDATE PersonsVec nếu cần
                if mv_feat is not None or mv_color is not None:
                    await self.db.execute(
                        """
                        UPDATE PersonsVec
                        SET feature_mean = vec_f32(?),
                            body_color_mean = vec_f32(?)
                        WHERE person_id = ?;
                        """,
                        (
                            mv_feat.tobytes() if mv_feat is not None else None,
                            mv_color.tobytes() if mv_color is not None else None,
                            person_id
                        )
                    )
                # 6) UPDATE FaceVector nếu cần
                if mv_face is not None:
                    async with self.db.execute("SELECT person_id FROM FaceVector WHERE person_id=?", (person_id,)) as cur:
                        exists = await cur.fetchone()
                    if exists:
                        await self.db.execute(
                            "UPDATE FaceVector SET face_embedding=vec_f32(?) WHERE person_id=?",
                            (mv_face.tobytes(), person_id)
                        )
                    else:
                        await self.db.execute(
                            "INSERT INTO FaceVector(person_id, face_embedding) VALUES(?, vec_f32(?))",
                            (person_id, mv_face.tobytes())
                        )

                # 7) UPDATE metadata nếu có
                fields, params = [], []
                if age is not None:
                    fields.append("age = ?"); params.append(age)
                if gender is not None:
                    fields.append("gender = ?"); params.append(gender)
                if race is not None:
                    fields.append("race = ?"); params.append(race)
                if fields:
                    sql = f"UPDATE PersonsMeta SET {', '.join(fields)} WHERE person_id = ?;"
                    params.append(person_id)
                    await self.db.execute(sql, params)

                # 8) Commit và checkpoint
                await self.db.commit()
                await self.db.execute("PRAGMA wal_checkpoint(FULL);")
                await self.db.commit()

            # 9) Cập nhật tracking_manager nếu có bbox_data và frame_id
            if bbox_data is not None and frame_id is not None:
                self.tracking_manager.update_track(person_id, bbox_data, frame_id, feature=feat_v)
                logger.info("Đã update ID trong tracking_manager")

        except Exception as e:
            logger.exception(f"[DB] Failed to update_person: {e}")
            return None
        
    async def match_id(self, new_gender, new_race, new_age, new_body_color, new_feature, new_face_embedding: None, bbox_data: None, frame_id: None):
        await self._ensure_db_ready()
        try:
            start = time.time()

            # 1. Check inputs
            if new_feature is None and new_body_color is None:
                logger.warning("No feature OR color given for matching.")
                return None

            # 2. Normalize feature (nếu có)
            nf = None
            if new_feature is not None:
                arr = np.asarray(new_feature, dtype=np.float32)
                nf = self.normalize(arr)

            # 3. Preprocess body color (nếu có) 
            nc = None
            if new_body_color is not None:
                bc = np.asarray(new_body_color, dtype=np.float32)
                if bc.shape != (17,3):
                    raise ValueError("new_body_color must be (17,3)")
                nc, _ = self.preprocess_color(bc)

            nface = self.normalize(np.array(new_face_embedding, dtype=np.float32)) if new_face_embedding is not None else None

            if bbox_data is not None and frame_id is not None:
                matched_pid = self.tracking_manager.match(bbox_data, frame_id, nf)
                if matched_pid is not None:
                    logger.info(f"MATCH ID with kalman filter: {matched_pid}")
                    # có thể thêm bước kiểm tra metadata/DB nếu muốn, hoặc chấp nhận ngay:
                    return matched_pid
                else:
                    logger.info("NO-MACTH_ID with kalman filter")

            async with self.db_lock:
                # 4. Metadata filter
                query = """
                    SELECT person_id FROM PersonsMeta
                    WHERE ((? IS NULL) OR (age = ? OR age IS NULL))
                    AND   ((? IS NULL) OR (gender = ? OR gender IS NULL))
                    AND   ((? IS NULL) OR (race = ? OR race IS NULL)) 
                """
                params = [new_age, new_age, new_gender, new_gender, new_race, new_race]
                cur = await self.db.execute(query, params)
                candidates = [row[0] for row in await cur.fetchall()]
                # logger.info(f"Metadata filter → {len(candidates)} candidates")
                if not candidates:
                    # logger.info(f"No candidates found for age={new_age}, gender={new_gender}, race={new_race}")
                    return None 

                # Bước 2: Ưu tiên face_embedding
                best_face_pid = None
                best_face_sim = 0.0
                if nface is not None:
                    placeholders = ",".join("?" for _ in candidates)
                    async with self.db.execute(
                        f"SELECT person_id, face_embedding FROM FaceVector WHERE person_id IN ({placeholders})",
                        candidates
                    ) as cur:
                        face_rows = await cur.fetchall()
                    if face_rows:
                        ids, face_vecs = [], []
                        for pid, buf in face_rows:
                            if buf:
                                vec = np.frombuffer(buf, dtype=np.float32)
                                if vec.ndim == 1:
                                    ids.append(pid)
                                    face_vecs.append(vec)
                                else:
                                    logger.error(f"[{self.device_id}] face_embedding cho {pid} không phải mảng 1 chiều")
                        if face_vecs:
                            face_vecs = np.stack(face_vecs, axis=0)
                            face_sims = cosine_similarity(face_vecs, nface.reshape(1, -1)).ravel()
                            best_idx = int(np.argmax(face_sims))
                            best_face_pid, best_face_sim = ids[best_idx], float(face_sims[best_idx])

                            if best_face_sim >= self.face_threshold:
                                logger.info(f"Khớp face_embedding {best_face_pid} với sim={best_face_sim:.4f} trong {time.time()-start:.2f}s")
                                return best_face_pid

                # 5. Body-color filter (nếu có), ngược lại cho qua hết
                if nc is not None:
                    placeholders = ",".join("?" for _ in candidates)
                    k = min(self.top_k, len(candidates))  # Thêm logic này
                    sql = f"""
                        SELECT person_id, distance FROM PersonsVec
                        WHERE person_id IN ({placeholders})
                        AND body_color_mean MATCH ? AND k = ?
                    """
                    cur = await self.db.execute(sql, candidates + [nc.tobytes(), k]) 
                    color_scores = [
                        (pid, 1 - dist)
                        for pid, dist in await cur.fetchall()
                        if 1 - dist > self.color_threshold
                    ]
                    # logger.info(f"Body color filter → {len(color_scores)} candidates above threshold")
                    if not color_scores:
                        return None
                    pids = [pid for pid, _ in color_scores]
                else:
                    pids = candidates


                # 6. Fetch feature vectors batch
                placeholders = ",".join("?" for _ in pids)
                cur = await self.db.execute(
                    f"SELECT person_id, feature_mean FROM PersonsVec WHERE person_id IN ({placeholders})",
                    pids
                )
                rows = await cur.fetchall()
                ids, vecs = zip(*[
                    (pid, np.frombuffer(buf, dtype=np.float32))
                    for pid, buf in rows
                ])
                ids = list(ids)
                vecs = np.stack(vecs, axis=0)
                if len(ids) < len(pids):
                    missing = set(pids) - set(ids)
                    logger.warning(f"Missing feature vectors for pids: {missing}") 

                # 8. Batch cosine similarity & select best
                sims = cosine_similarity(vecs, nf.reshape(1, -1)).ravel()  # luôn trả về 1-d array
                best_idx = int(np.argmax(sims))
                best_pid, best_sim = ids[best_idx], float(sims[best_idx])

                if best_sim < self.feature_threshold:
                    logger.info(f"Best pid={best_pid} with sim={best_sim:.4f} but below feature_threshold={self.feature_threshold}")
                    return None

                logger.info(f"Selected pid={best_pid} with sim={best_sim:.4f}") 
                # logger.info(f"match_id took {time.time()-start:.2f}s")
                return best_pid
        except Exception as e:
            logger.exception(f"[DB] Failed to match_id: {e}")
            return None

    # async def match_id(self, new_gender, new_race, new_age, new_body_color, new_feature):
    #     await self._ensure_db_ready()
    #     try:
    #         start = time.time()

    #         # 1. Check inputs
    #         if new_feature is None and new_body_color is None:
    #             logger.warning("No feature OR color given for matching.")
    #             return None

    #         # 2. Normalize feature (nếu có)
    #         nf = None
    #         if new_feature is not None:
    #             arr = np.asarray(new_feature, dtype=np.float32)
    #             if arr.shape != (512,):
    #                 raise ValueError("new_feature must be 512-dim")
    #             nf = self.normalize(arr)

    #         # 3. Preprocess body color (nếu có) 
    #         nc = None
    #         if new_body_color is not None:
    #             bc = np.asarray(new_body_color, dtype=np.float32)
    #             if bc.shape != (17,3):
    #                 raise ValueError("new_body_color must be (17,3)")
    #             nc, _ = self.preprocess_color(bc)

    #         async with self.db_lock:
    #             # 4. Metadata filter
    #             query = """
    #                 SELECT person_id FROM PersonsMeta
    #                 WHERE ((? IS NULL) OR (age = ? OR age IS NULL))
    #                 AND   ((? IS NULL) OR (gender = ? OR gender IS NULL))
    #                 AND   ((? IS NULL) OR (race = ? OR race IS NULL)) 
    #             """
    #             params = [new_age, new_age, new_gender, new_gender, new_race, new_race]
    #             cur = await self.db.execute(query, params)
    #             candidates = [row[0] for row in await cur.fetchall()]
    #             # logger.info(f"Metadata filter → {len(candidates)} candidates")
    #             if not candidates:
    #                 # logger.info(f"No candidates found for age={new_age}, gender={new_gender}, race={new_race}")
    #                 return None 


    #             # 5. Body-color filter (nếu có), ngược lại cho qua hết
    #             if nc is not None:
    #                 placeholders = ",".join("?" for _ in candidates)
    #                 k = min(self.top_k, len(candidates))  # Thêm logic này
    #                 sql = f"""
    #                     SELECT person_id, distance FROM PersonsVec
    #                     WHERE person_id IN ({placeholders})
    #                     AND body_color_mean MATCH ? AND k = ?
    #                 """
    #                 cur = await self.db.execute(sql, candidates + [nc.tobytes(), k]) 
    #                 color_scores = [
    #                     (pid, 1 - (dist**2)/2)
    #                     for pid, dist in await cur.fetchall()
    #                     if 1 - (dist**2)/2 > self.color_threshold
    #                 ]
    #                 # logger.info(f"Body color filter → {len(color_scores)} candidates above threshold")
    #                 if not color_scores:
    #                     return None
    #                 pids = [pid for pid, _ in color_scores]
    #             else:
    #                 pids = candidates


    #             # 6. Fetch feature vectors batch
    #             placeholders = ",".join("?" for _ in pids)
    #             cur = await self.db.execute(
    #                 f"SELECT person_id, feature_mean FROM PersonsVec WHERE person_id IN ({placeholders})",
    #                 pids
    #             )
    #             rows = await cur.fetchall()
    #             ids, vecs = zip(*[
    #                 (pid, np.frombuffer(buf, dtype=np.float32))
    #                 for pid, buf in rows
    #             ])
    #             ids = list(ids)
    #             vecs = np.stack(vecs, axis=0)
    #             if len(ids) < len(pids):
    #                 missing = set(pids) - set(ids)
    #                 logger.warning(f"Missing feature vectors for pids: {missing}") 

    #             # 8. Batch cosine similarity & select best
    #             sims = cosine_similarity(vecs, nf.reshape(1, -1)).ravel()  # luôn trả về 1-d array
    #             best_idx = int(np.argmax(sims))
    #             best_pid, best_sim = ids[best_idx], float(sims[best_idx])

    #             if best_sim < self.feature_threshold:
    #                 logger.info(f"Best pid={best_pid} with sim={best_sim:.4f} but below feature_threshold={self.feature_threshold}")
    #                 return None

    #             # logger.info(f"Selected pid={best_pid} with sim={best_sim:.4f}") 
    #             # logger.info(f"match_id took {time.time()-start:.2f}s")
    #             return best_pid
    #     except Exception as e:
    #         logger.exception(f"[DB] Failed to match_id: {e}")
    #         return None

    async def remove_low_person_ids(self, ids_to_remove: list[int]):
        """Xoá các person_id được chỉ định khỏi DB và bộ nhớ."""
        try:
            if not ids_to_remove:
                logger.info("No person IDs to remove.")
                return

            async with self.db_lock:
                await self.db.execute("PRAGMA foreign_keys = ON;")
                placeholders = ",".join("?" * len(ids_to_remove))
                await self.db.execute(
                    f"DELETE FROM Persons WHERE person_id IN ({placeholders});",
                    ids_to_remove
                )
                await self.db.commit()

                # Cleanup in-memory histories
                for pid in ids_to_remove:
                    self.feature_history.pop(pid, None)
                    self.color_history.pop(pid, None)
                
                logger.info(f"[DB] Removed {len(ids_to_remove)} person IDs: {ids_to_remove}")
        except Exception as e:
            logger.exception(f"[DB] Failed to remove_low_person_ids: {e}")
            return None  
        
    async def replace_temp_person_id(self, temp_id: str, match_id: str) -> bool:
        await self._ensure_db_ready()

        # Cho phép phần còn lại của hệ thống dùng ngay alias mới
        self.alias_map[temp_id] = match_id

        async with self.db_lock:
            try:
                logger.info("[DB] Replace temp_id=%s → match_id=%s", temp_id, match_id)

                # 1) Đảm bảo match_id tồn tại trong Persons
                await self.db.execute(
                    "INSERT OR IGNORE INTO Persons(person_id) VALUES (?);",
                    (match_id,)
                )

                # 2) Các bảng KHÔNG có UNIQUE/PK trên person_id → update trực tiếp
                for tbl in ("Detections", "Appearance", "PublishStatus"):
                    await self.db.execute(
                        f"UPDATE {tbl} SET person_id=? WHERE person_id=?;",
                        (match_id, temp_id)
                    )

                # 3) PersonsMeta có PK ⇒ phải xoá trước khi chuyển
                await self.db.execute(
                    "DELETE FROM PersonsMeta WHERE person_id=?;",   # xoá (nếu) bản ghi match_id đã có
                    (match_id,)
                )
                await self.db.execute(
                    "UPDATE PersonsMeta SET person_id=? WHERE person_id=?;",  # chuyển temp_id sang match_id
                    (match_id, temp_id)
                )

                # 4) Vector: copy & xoá
                cur = await self.db.execute(
                    "SELECT feature_mean, body_color_mean FROM PersonsVec WHERE person_id=?;",
                    (temp_id,)
                )
                if (vec := await cur.fetchone()):
                    await self.db.execute(
                        "INSERT OR REPLACE INTO PersonsVec(person_id, feature_mean, body_color_mean) "
                        "VALUES(?, ?, ?);",
                        (match_id, vec[0], vec[1]) 
                    )
                    await self.db.execute(
                        "DELETE FROM PersonsVec WHERE person_id=?;", 
                        (temp_id,)
                    )

                # 5) Xoá temp_id khỏi Persons
                await self.db.execute(
                    "DELETE FROM Persons WHERE person_id=?;",
                    (temp_id,)
                )

                await self.db.commit()  

            except Exception as e:
                await self.db.rollback()
                logger.exception("[DB] Failed replace temp_id=%s → match_id=%s : %s",
                                temp_id, match_id, e)
                return False

        # 6) Gộp lịch sử RAM
        if temp_id in self.feature_history:
            self.feature_history[match_id].extend(self.feature_history.pop(temp_id))
        if temp_id in self.color_history:
            self.color_history[match_id].extend(self.color_history.pop(temp_id))

        logger.info("[DB] Replace temp_id=%s → match_id=%s : DONE", temp_id, match_id)
        return True

    async def merge_temp_person_id(self, temp_id: str, closest_id: str) -> bool:
        """
        Merge temp_id into closest_id when metadata matches.
        - Metadata (age, gender, race) must be identical or None.
        - Update Detections and alias_map.
        - Merge feature and color history in RAM instead of DB vectors.
        - Remove temp_id entries.
        """
        await self._ensure_db_ready()
        # Record alias for in-memory checks
        self.alias_map[temp_id] = closest_id

        async with self.db_lock:
            try:
                logger.info("[DB] Merge temp_id=%s → closest_id=%s", temp_id, closest_id)

                # 1) Fetch and compare metadata
                cur = await self.db.execute(
                    "SELECT person_id, age, gender, race FROM PersonsMeta WHERE person_id IN (?, ?);",
                    (temp_id, closest_id)
                )
                rows = await cur.fetchall()
                meta = {row[0]: row[1:] for row in rows}
                if temp_id not in meta or closest_id not in meta:
                    logger.warning("[DB] Missing PersonsMeta for %s or %s", temp_id, closest_id)
                    return False
                for field_temp, field_closest in zip(meta[temp_id], meta[closest_id]):
                    if field_temp is not None and field_closest is not None and field_temp != field_closest:
                        logger.warning(
                            "[DB] Metadata mismatch — cannot merge: %s vs %s",
                            meta[temp_id], meta[closest_id]
                        )
                        return False

                # 2) Update all detections to point to the merged ID
                await self.db.execute(
                    "UPDATE Detections SET person_id=? WHERE person_id=?;",
                    (closest_id, temp_id)
                )

                # 3) Remove temp_id entries from PersonsMeta and Persons
                await self.db.execute(
                    "DELETE FROM PersonsMeta WHERE person_id=?;", (temp_id,)
                )
                await self.db.execute(
                    "DELETE FROM Persons WHERE person_id=?;", (temp_id,)
                )

                # 4) Remove temp_id from PersonsVec
                await self.db.execute(
                    "DELETE FROM PersonsVec WHERE person_id=?;", (temp_id,)
                )

                await self.db.commit()
                logger.info("[DB] Successfully merged %s into %s", temp_id, closest_id)

            except Exception as e:
                await self.db.rollback()
                logger.exception(
                    "[DB] Failed to merge temp_id=%s → closest_id=%s: %s",
                    temp_id, closest_id, e
                )
                return False

        # Merge in-memory feature and color histories (only after DB success)
        if temp_id in self.feature_history:
            self.feature_history.setdefault(closest_id, [])
            self.feature_history[closest_id].extend(self.feature_history.pop(temp_id))
        if temp_id in self.color_history:
            self.color_history.setdefault(closest_id, [])
            self.color_history[closest_id].extend(self.color_history.pop(temp_id))

        return True

    async def find_closest_by_id(self, temp_id: str) -> tuple[str | None, float]:
        """
        Tìm person_id gần nhất với temp_id dựa trên feature_history trong RAM.
        Dùng batch cosine_similarity từ sklearn để tăng tốc.
        """
        history = self.feature_history.get(temp_id)
        if not history:
            logger.warning(f"[find_closest_by_id] Không có history cho temp_id={temp_id}")
            return None, 0.0

        # Query vector: mean + normalize
        q_vec = np.mean(history, axis=0).astype(np.float32)  # shape (D,)
        q_vec /= np.linalg.norm(q_vec) + 1e-8

        # Tạo list tất cả vector & id khác temp_id
        ids, vectors = [], []
        for pid, feats in self.feature_history.items():
            if pid == temp_id or not feats:
                continue
            vec = np.mean(feats, axis=0).astype(np.float32)  # shape (D,)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            vectors.append(vec / norm)
            ids.append(pid)

        if not vectors:
            logger.warning(f"[find_closest_by_id] Không tìm được neighbor trong RAM cho temp_id={temp_id}")
            return None, 0.0
        
        vectors = np.stack(vectors, axis=0)  # shape (N, D)
        # Batch cosine similarity
        sims = cosine_similarity([q_vec], vectors)[0]
        best_idx = int(np.argmax(sims))
        return ids[best_idx], float(sims[best_idx]) 


def compute_ema(history: deque) -> np.ndarray | None:
    n = len(history)
    if n == 0:
        return None
    alpha = 2.0 / (n + 1)
    ema = history[0].astype(np.float32)
    for x in list(history)[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema