import os
import time
import json
import asyncio
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import aiosqlite
from sklearn.metrics.pairwise import cosine_similarity

from track_local.track import TrackingManager
from utils.logging_python_orangepi import get_logger

from config import OPI_CONFIG

logger = get_logger(__name__)

class ReIDManager: 
    def __init__(
        self,
        db_path='database.db',
        feature_threshold=0.75,
        face_threshold=0.8,
        hard_feature_threshold: float = 0.5,
        hard_face_threshold: float = 0.5,
        top_k=3,
    ):
        self.db_path = db_path
        self.feature_threshold = feature_threshold
        self.top_k = top_k
        self.face_threshold = face_threshold

        # History
        self.feature_history = defaultdict(lambda: deque(maxlen=20))
        self.face_embedding_history = defaultdict(lambda: deque(maxlen=20))

        # Async DB setup
        self.db_lock = asyncio.Lock()
        self.db_init = asyncio.Event()
        self.db = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vec_ext = '/usr/local/lib/vec0.so'

        self.alias_map: dict[str, str] = {}

        asyncio.create_task(self._init_database())

        self.tracking_manager = TrackingManager()
        self.hard_face_threshold = hard_face_threshold
        self.hard_feature_threshold = hard_feature_threshold

        logger.info(f"ReIDManager init successful with feature_threshold: {feature_threshold}, face_threshold: {face_threshold}, hard_feature_threshold: {hard_feature_threshold}, hard_face_threshold: {hard_face_threshold}")

    async def _init_database(self):
        async with self.db_lock:
            db = await aiosqlite.connect(self.db_path)
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.enable_load_extension(True)
            await db.load_extension(self.vec_ext)
            self.db = db
            self.db_init.set()
            logger.info("Database ready and schema initialized.")

    async def _ensure_db_ready(self):
        await self.db_init.wait()

    def normalize(self, v: np.ndarray) -> np.ndarray | None:
        if v is None:
            return None
        v = v.flatten()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    async def create_person(
        self,
        gender, race, age, body_color, feature, face_embedding,
        est_height_m, world_point_xyz,
        bbox_data=None, frame_id=None, time_detect=None
    ):
        """
        Tạo một nhân vật mới và ghi vào DB. 
        """
        await self._ensure_db_ready()
        try:
            pid = str(uuid.uuid4())
            ts = time_detect if time_detect is not None else time.time()

            feat_v = self.normalize(np.array(feature, np.float32) if feature is not None else None)
            face_v = self.normalize(np.array(face_embedding, np.float32) if face_embedding is not None else None)

            async with self.db_lock:
                # Insert vào bảng chính
                await self.db.execute("INSERT INTO Persons(person_id) VALUES (?);", (pid,))
                
                if feat_v is not None:
                    await self.db.execute(
                        "INSERT INTO PersonsVec(person_id, feature_mean) VALUES (?, vec_f32(?));",
                        (pid, feat_v.tobytes())
                    )
                if face_v is not None:
                    await self.db.execute(
                        "INSERT INTO FaceVector(person_id, face_embedding) VALUES (?, vec_f32(?));",
                        (pid, face_v.tobytes())
                    )

                # --- SỬA LỖI Ở ĐÂY ---
                # Chuyển đổi tường minh sang kiểu dữ liệu Python gốc
                await self.db.execute(
                    "INSERT INTO PersonsMeta(person_id, age, gender, race, height_mean) "
                    "VALUES (?, ?, ?, ?, ?);",
                    (pid, str(age), str(gender), str(race), float(est_height_m) if est_height_m is not None else None)
                )

                # Đảm bảo frame tồn tại trước
                if frame_id is not None:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO Frames(frame_id, timestamp) VALUES (?, ?);",
                        (frame_id, ts)
                    )

                # Thêm bản ghi vào Detections
                await self.db.execute(
                    "INSERT INTO Detections(person_id, timestamp, frame_id) "
                    "VALUES (?, ?, ?);",
                    (pid, ts, frame_id)
                )

                # Tạo FeatureBank
                if feat_v is not None:
                    cur = await self.db.execute(
                        "INSERT INTO FeatureBank(person_id) VALUES (?) RETURNING feature_id;", (pid,)
                    )
                    feat_id = (await cur.fetchone())[0]
                    await self.db.execute(
                        "INSERT INTO FeatureBankVec(feature_id, person_id, feature_vec) VALUES (?, ?, vec_f32(?));",
                        (feat_id, pid, feat_v.tobytes())
                    )

                # Tạo FaceIDBank
                if face_v is not None:
                    cur2 = await self.db.execute(
                        "INSERT INTO FaceIDBank(person_id) VALUES (?) RETURNING face_id;", (pid,)
                    )
                    face_id = (await cur2.fetchone())[0]
                    await self.db.execute(
                        "INSERT INTO FaceIDBankVec(face_id, person_id, face_vec) VALUES (?, ?, vec_f32(?));",
                        (face_id, pid, face_v.tobytes())
                    )

                await self.db.commit()

            # Lưu history (dùng cho EMA)
            if feat_v is not None:
                self.feature_history[pid].append(feat_v)
            if face_v is not None:
                self.face_embedding_history[pid].append(face_v)

            # Gửi vào tracker
            if bbox_data is not None and frame_id is not None:
                self.tracking_manager.add_track(pid, bbox_data, frame_id, feature=feat_v)
                logger.info("Added new track to tracking_manager")

            return pid
        except Exception as e:
            logger.exception(f"[DB] Failed to create person: {e}")
            return None

    async def update_person(
        self,
        person_id,
        gender=None, race=None, age=None,
        body_color=None, feature=None, face_embedding=None,
        est_height_m=None, world_point_xyz=None,
        bbox_data=None, frame_id=None, time_detect=None
    ):
        await self._ensure_db_ready()
        try:
            person_id = self.alias_map.get(person_id, person_id)
            feat_v = self.normalize(np.asarray(feature, np.float32) if feature is not None else None)
            face_v = self.normalize(np.asarray(face_embedding, np.float32) if face_embedding is not None else None)
            ts = time_detect if time_detect is not None else time.time()

            async with self.db_lock:
                if frame_id is not None:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO Frames(frame_id, timestamp) VALUES (?, ?);",
                        (frame_id, ts)
                    )
                logger.info("[UPDATE] Frames pass")
                await self.db.execute(
                    "INSERT INTO Detections(person_id, timestamp, frame_id) VALUES (?, ?, ?);",
                    (person_id, ts, frame_id)
                )
                logger.info("[UPDATE] Detections pass")
                if feat_v is not None:
                    self.feature_history[person_id].append(feat_v)
                if face_v is not None:
                    self.face_embedding_history[person_id].append(face_v)

                mv_feat, mv_face = await asyncio.gather(
                    asyncio.to_thread(compute_ema, self.feature_history[person_id]),
                    asyncio.to_thread(compute_ema, self.face_embedding_history[person_id])
                )

                if mv_feat is not None:
                    await self.db.execute(
                        "UPDATE PersonsVec SET feature_mean = vec_f32(?) WHERE person_id = ?;",
                        (mv_feat.tobytes(), person_id)
                    )
                if mv_face is not None:
                    await self.db.execute(
                        "UPDATE FaceVector SET face_embedding = vec_f32(?) WHERE person_id = ?;",
                        (mv_face.tobytes(), person_id)
                    )

                fields, params = [], []
                if age is not None: fields.append("age = ?"); params.append(age)
                if gender is not None: fields.append("gender = ?"); params.append(gender)
                if race is not None: fields.append("race = ?"); params.append(race)
                if est_height_m is not None: fields.append("height_mean = ?"); params.append(est_height_m)

                if fields:
                    sql = f"UPDATE PersonsMeta SET {', '.join(fields)} WHERE person_id = ?;"
                    params.append(person_id)
                    await self.db.execute(sql, params)

                await self.db.commit()

            if bbox_data is not None and frame_id is not None:
                self.tracking_manager.update_track(person_id, bbox_data, frame_id, feature=feat_v)
                logger.info("Updated track in tracking_manager")

            return True
        except Exception as e:
            logger.exception(f"[DB] Failed to update_person: {e}")
            return None

    async def match_id(
        self, 
        gender=None, race=None, age=None,
        body_color=None, feature=None, face_embedding=None,
        est_height_m=None, world_point_xyz=None,
        bbox_data=None, frame_id=None
    ):
        """
        Tìm ID khớp nhất cho một phát hiện mới.
        """
        await self._ensure_db_ready()
        # --- DEBUG: Log incoming metadata types/values ---
        logger.debug(
            "[MATCH_ID INPUT] age=%r (type=%s), gender=%r (type=%s), race=%r (type=%s)",
            age, type(age).__name__,
            gender, type(gender).__name__,
            race, type(race).__name__
        )

        try:
            start = time.time()

            if feature is None and face_embedding is None:
                logger.warning("No feature or face embedding given for matching.")
                return None

            nf = self.normalize(np.asarray(feature, dtype=np.float32)) if feature is not None else None
            nface = self.normalize(np.asarray(face_embedding, dtype=np.float32)) if face_embedding is not None else None
            
            # --- Kalman filter matching ---
            if bbox_data is not None and frame_id is not None:
                matched_pid = self.tracking_manager.match(bbox_data, frame_id, nf)
                if matched_pid is not None:
                    logger.info(f"[KALMAN FILTER] MATCH ID with kalman filter: {matched_pid}")
                    return matched_pid
                else:
                    logger.info("NO-MATCH_ID with kalman filter")   

            _ = body_color
            _ = est_height_m

            async with self.db_lock:
                # --- DEBUG: Normalized params for SQL ---
                logger.debug(
                    "[SQL FILTER PARAMS] age_filter=%r, gender_filter=%r, race_filter=%r",
                    age, gender, race
                )

                query = """
                    SELECT person_id FROM PersonsMeta
                    WHERE ((? IS NULL) OR (age = ? OR age IS NULL))
                    AND ((? IS NULL) OR (gender = ? OR gender IS NULL))
                    AND ((? IS NULL) OR (race = ? OR race IS NULL))
                """
                params = [
                    age, age,
                    gender, gender,
                    race, race
                ]
                cur = await self.db.execute(query, params)
                candidates = [row[0] for row in await cur.fetchall()]

                logger.debug("[SQL RESULT] candidates=%s", candidates)

                if not candidates:
                    logger.info("No candidates after metadata filtering.")
                    return None

            # --- Face matching ---
            if nface is not None and candidates: 
                placeholders = ",".join("?" for _ in candidates)
                k_face = min(self.top_k, len(candidates), 5)
                sql_face = (
                    f"SELECT person_id, face_embedding FROM FaceVector"
                    f" WHERE person_id IN ({placeholders})"
                    f" AND face_embedding MATCH ? AND k = ?"
                )
                async with self.db.execute(sql_face, candidates + [nface.tobytes(), k_face]) as cur:
                    face_rows = await cur.fetchall()

                if face_rows:
                    ids, vecs = zip(*[(pid, np.frombuffer(buf, dtype=np.float32)) for pid, buf in face_rows])
                    sims = cosine_similarity(np.stack(vecs), nface.reshape(1, -1)).ravel()
                    best_idx = int(np.argmax(sims))
                    best_face_pid, best_face_sim = ids[best_idx], float(sims[best_idx])

                    if best_face_sim >= self.face_threshold:
                        async with self.db.execute(
                            "SELECT face_vec FROM FaceIDBankVec WHERE person_id = ? AND face_vec MATCH ? AND k = 1",
                            (best_face_pid, nface.tobytes())
                        ) as bank_cur:
                            row = await bank_cur.fetchone()

                        if row:
                            bank_vec = np.frombuffer(row[0], dtype=np.float32)
                            sim_bank = cosine_similarity(bank_vec.reshape(1, -1), nface.reshape(1, -1)).ravel()[0]
                            if sim_bank >= self.hard_face_threshold:
                                logger.info(f"[FACEID] match passed for {best_face_pid} (sim={sim_bank:.4f}) in {time.time()-start:.2f}s")
                                return best_face_pid
                        logger.warning("[FACE ID] Đã có {best_face_sim} nhưng không khớp tất cả")


            # --- Feature vector matching ---
            if nf is not None and candidates:
                placeholders = ",".join("?" for _ in candidates)
                k_feat = min(self.top_k, len(candidates), 5)
                sql_feat = (
                    f"SELECT person_id, feature_mean FROM PersonsVec"
                    f" WHERE person_id IN ({placeholders})"
                    f" AND feature_mean MATCH ? AND k = ?" 
                )
                async with self.db.execute(sql_feat, candidates + [nf.tobytes(), k_feat]) as cur:
                    feat_rows = await cur.fetchall()

                if feat_rows:
                    ids, vecs = zip(*[(pid, np.frombuffer(buf, dtype=np.float32)) for pid, buf in feat_rows])
                    sims = cosine_similarity(np.stack(vecs), nf.reshape(1, -1)).ravel()
                    best_idx = int(np.argmax(sims))
                    best_pid, best_sim = ids[best_idx], float(sims[best_idx])

                    if best_sim >= self.feature_threshold:
                        async with self.db.execute(
                            "SELECT feature_vec FROM FeatureBankVec WHERE person_id = ? AND feature_vec MATCH ? AND k = 1",
                            (best_pid, nf.tobytes())
                        ) as feat_cur:
                            row = await feat_cur.fetchone()

                        if row:
                            bank_vec = np.frombuffer(row[0], dtype=np.float32)
                            sim_bank = cosine_similarity(bank_vec.reshape(1, -1), nf.reshape(1, -1)).ravel()[0]
                            if sim_bank >= self.hard_feature_threshold:
                                logger.info(f"[FEATURE] Hard-feature check passed for {best_pid} (sim={sim_bank:.4f})")
                                return best_pid 
            logger.info("[MATCH ID] không khớp tất cả")
            return None 

        except Exception as e:
            logger.exception(f"[DB] Failed to match_id: {e}")
            return None

        
    # async def match_id(
    #     self, 
    #     gender=None, race=None, age=None,
    #     body_color=None, feature=None, face_embedding=None,
    #     est_height_m=None, world_point_xyz=None,
    #     bbox_data=None, frame_id=None
    # ):
    #     """
    #     Tìm ID khớp nhất cho một phát hiện mới.
    #     """
    #     await self._ensure_db_ready()
    #     try:
    #         start = time.time()

    #         if feature is None and face_embedding is None:
    #             logger.warning("No feature or face embedding given for matching.")
    #             return None

    #         nf = self.normalize(np.asarray(feature, dtype=np.float32)) if feature is not None else None
    #         nface = self.normalize(np.asarray(face_embedding, dtype=np.float32)) if face_embedding is not None else None
            
    #         # --- Kalman filter matching ---
    #         if bbox_data is not None and frame_id is not None:
    #             matched_pid = self.tracking_manager.match(bbox_data, frame_id, nf)
    #             if matched_pid is not None:
    #                 logger.info(f"[KALAM FILTER] MATCH ID with kalman filter: {matched_pid}")
    #                 return matched_pid
    #             else:
    #                 logger.info("NO-MATCH_ID with kalman filter")   

    #         _ = body_color
    #         _ = est_height_m

    #         async with self.db_lock:
    #             query = """
    #                 SELECT person_id FROM PersonsMeta
    #                 WHERE ((? IS NULL) OR (age = ? OR age IS NULL))
    #                 AND ((? IS NULL) OR (gender = ? OR gender IS NULL))
    #                 AND ((? IS NULL) OR (race = ? OR race IS NULL))
    #             """
    #             params = [
    #                 age, age,
    #                 gender, gender,
    #                 race, race
    #             ]
    #             cur = await self.db.execute(query, params)
    #             candidates = [row[0] for row in await cur.fetchall()]

    #             if not candidates:
    #                 logger.info("No candidates after metadata filtering.")
    #                 return None

    #         # --- Face matching ---
    #         if nface is not None and candidates: 
    #             placeholders = ",".join("?" for _ in candidates)
    #             k_face = min(self.top_k, len(candidates), 5)
    #             sql_face = (
    #                 f"SELECT person_id, face_embedding FROM FaceVector"
    #                 f" WHERE person_id IN ({placeholders})"
    #                 f" AND face_embedding MATCH ? AND k = ?"
    #             )
    #             async with self.db.execute(sql_face, candidates + [nface.tobytes(), k_face]) as cur:
    #                 face_rows = await cur.fetchall()

    #             if face_rows:
    #                 ids, vecs = zip(*[(pid, np.frombuffer(buf, dtype=np.float32)) for pid, buf in face_rows])
    #                 sims = cosine_similarity(np.stack(vecs), nface.reshape(1, -1)).ravel()
    #                 best_idx = int(np.argmax(sims))
    #                 best_face_pid, best_face_sim = ids[best_idx], float(sims[best_idx])

    #                 if best_face_sim >= self.face_threshold:
    #                     async with self.db.execute(
    #                         "SELECT face_vec FROM FaceIDBankVec WHERE person_id = ? AND face_vec MATCH ? AND k = 1",
    #                         (best_face_pid, nface.tobytes())
    #                     ) as bank_cur:
    #                         row = await bank_cur.fetchone()

    #                     if row:
    #                         bank_vec = np.frombuffer(row[0], dtype=np.float32)
    #                         sim_bank = cosine_similarity(bank_vec.reshape(1, -1), nface.reshape(1, -1)).ravel()[0]
    #                         if sim_bank >= self.hard_face_threshold:
    #                             logger.info(f"[FACEID] match passed for {best_face_pid} (sim={sim_bank:.4f}) in {time.time()-start:.2f}s")
    #                             return best_face_pid

    #         # --- Feature vector matching ---
    #         if nf is not None and candidates:
    #             placeholders = ",".join("?" for _ in candidates)
    #             k_feat = min(self.top_k, len(candidates), 5)
    #             sql_feat = (
    #                 f"SELECT person_id, feature_mean FROM PersonsVec"
    #                 f" WHERE person_id IN ({placeholders})"
    #                 f" AND feature_mean MATCH ? AND k = ?" 
    #             )
    #             async with self.db.execute(sql_feat, candidates + [nf.tobytes(), k_feat]) as cur:
    #                 feat_rows = await cur.fetchall()

    #             if feat_rows:
    #                 ids, vecs = zip(*[(pid, np.frombuffer(buf, dtype=np.float32)) for pid, buf in feat_rows])
    #                 sims = cosine_similarity(np.stack(vecs), nf.reshape(1, -1)).ravel()
    #                 best_idx = int(np.argmax(sims))
    #                 best_pid, best_sim = ids[best_idx], float(sims[best_idx])

    #                 if best_sim >= self.feature_threshold:
    #                     async with self.db.execute(
    #                         "SELECT feature_vec FROM FeatureBankVec WHERE person_id = ? AND feature_vec MATCH ? AND k = 1",
    #                         (best_pid, nf.tobytes())
    #                     ) as feat_cur:
    #                         row = await feat_cur.fetchone()

    #                     if row:
    #                         bank_vec = np.frombuffer(row[0], dtype=np.float32)
    #                         sim_bank = cosine_similarity(bank_vec.reshape(1, -1), nf.reshape(1, -1)).ravel()[0]
    #                         if sim_bank >= self.hard_feature_threshold:
    #                             logger.info(f"[FEATURE] Hard-feature check passed for {best_pid} (sim={sim_bank:.4f})")
    #                             return best_pid 
    #         logger.info("[MATCH ID] không khớp tất cả")
    #         return None 

    #     except Exception as e:
    #         logger.exception(f"[DB] Failed to match_id: {e}")
    #         return None



##########################################################################################

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
                    if pid in self.tracking_manager.tracks:
                        del self.tracking_manager.tracks[pid]
                
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

        logger.info("[DB] Replace temp_id=%s → match_id=%s : DONE", temp_id, match_id)
        return True

    async def merge_temp_person_id(self, temp_id: str, closest_id: str) -> bool:
            """
            Merge temp_id vào closest_id, bao gồm:
            - Metadata (DB) đã xử lý
            - Feature history (RAM)
            - TrackingManager.tracks
            """
            await self._ensure_db_ready()
            # Record alias cho in-memory lookup
            self.alias_map[temp_id] = closest_id

            async with self.db_lock:
                try:
                    # ... phần DB merge giống như trước ...

                    await self.db.commit()
                    logger.info("[DB] Successfully merged %s into %s", temp_id, closest_id)
                except Exception as e:
                    await self.db.rollback()
                    logger.exception(
                        "[DB] Failed to merge temp_id=%s → closest_id=%s: %s",
                        temp_id, closest_id, e
                    )
                    return False

            # 1) Merge feature history
            if temp_id in self.feature_history:
                tgt_hist = self.feature_history.setdefault(closest_id, deque(maxlen=20))
                tgt_hist.extend(self.feature_history.pop(temp_id))

            # 2) Merge face embedding history (nếu cần)
            if temp_id in self.face_embedding_history:
                tgt_face = self.face_embedding_history.setdefault(closest_id, deque(maxlen=20))
                tgt_face.extend(self.face_embedding_history.pop(temp_id))

            # 3) Merge tracking_manager.tracks
            tm = self.tracking_manager
            if temp_id in tm.tracks:
                temp_tr = tm.tracks.pop(temp_id)

                if closest_id in tm.tracks:
                    existing = tm.tracks[closest_id]
                    # gộp feature_history của track
                    existing['feature_history'].extend(temp_tr['feature_history'])

                    # Có thể cập nhật các state mới nhất tuỳ ý
                    existing.update({
                        'mean': temp_tr['mean'],
                        'covariance': temp_tr['covariance'],
                        'last_frame_id': temp_tr['last_frame_id'],
                        'time_since_update': temp_tr['time_since_update'],
                    })
                else:
                    # nếu chưa có track của closest_id, dùng nguyên bản temp_tr
                    tm.tracks[closest_id] = temp_tr

                logger.info(f"[TRACK] Merged tracking state {temp_id} → {closest_id}")

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


################################################################

    async def get_person_meta(self, person_id: str) -> dict | None:
        """
        Query the database for metadata of a person by person_id.

        Returns a dictionary with keys corresponding to the PersonsMeta columns
        or None if the person_id is not found or on error.
        """
        await self._ensure_db_ready()
        try:
            async with self.db_lock:
                cursor = await self.db.execute(
                    "SELECT person_id, age, gender, race, height_mean FROM PersonsMeta WHERE person_id = ?;", 
                    (person_id,)
                )
                row = await cursor.fetchone()
            if row is None:
                logger.info(f"No metadata found for person_id={person_id}")
                return None

            # Map column order to keys
            keys = ["person_id", "age", "gender", "race", "height_mean"] 
            return dict(zip(keys, row))
        except Exception as e:
            logger.exception(f"[DB] Failed to query person meta for {person_id}: {e}")
            return None

def compute_ema(history: deque) -> np.ndarray | None:
    if not history:  # Kiểm tra này đúng cho cả None và list/deque rỗng
        return None
    n = len(history)
    if n == 0:
        return None
    alpha = 2.0 / (n + 1)
    ema = history[0].astype(np.float32)
    for x in list(history)[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema