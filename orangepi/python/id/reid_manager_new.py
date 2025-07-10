import os
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

import numpy as np
import aiosqlite
import uuid
# import sqlite_vec
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

        self.tracking_manager = TrackingManager()

        self.hard_face_threshold = 0.85
        self.hard_feature_threshold = 0.8

        # asyncio.create_task(self.setup_rabbitmq(rabbitmq_url))
        logger.info("ReIDManager init successful")

    async def _init_database(self):
        async with self.db_lock: 
            # 2) Mở kết nối
            db = await aiosqlite.connect(self.db_path)
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.enable_load_extension(True)
            await db.load_extension(self.vec_ext) 
            # await db.load_extension(sqlite_vec.loadable_path())

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

    def normalize(self, v: np.ndarray) -> np.ndarray | None:
        """ 
        Chuẩn hóa một vector. Trả về None nếu đầu vào là None.
        """
        # ### <<< THAY ĐỔI: Thêm câu lệnh kiểm tra này
        if v is None:
            return None
        # ### <<< KẾT THÚC THAY ĐỔI

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
                            est_height_m, head_point_3d,
                            bbox_data=None, frame_id=None):
        await self._ensure_db_ready()
        try:
            feat_v = self.normalize(np.array(feature, np.float32) if feature is not None else None)
            face_v = self.normalize(np.array(face_embedding, np.float32) if face_embedding is not None else None)
            _ = body_color 
            pid = str(uuid.uuid4())
            ts = asyncio.get_event_loop().time()

            async with self.db_lock:
                await self.db.execute("INSERT INTO Persons(person_id) VALUES (?);", (pid,))
                
                feat_bytes = feat_v.tobytes() if feat_v is not None else None
                await self.db.execute("INSERT INTO PersonsVec(person_id, feature_mean) VALUES (?, vec_f32(?));", (pid, feat_bytes))

                if face_v is not None:
                    await self.db.execute("INSERT OR REPLACE INTO FaceVector(person_id, face_embedding) VALUES (?, vec_f32(?));", (pid, face_v.tobytes()))
                
                await self.db.execute("INSERT INTO PersonsMeta(person_id, age, gender, race, height_mean) VALUES (?, ?, ?, ?, ?);", (pid, age, gender, race, est_height_m))

                # ### <<< SỬA LỖI TẠI ĐÂY
                # Đảm bảo frame_id tồn tại trong bảng Frames trước khi tham chiếu đến nó.
                if frame_id is not None:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO Frames(frame_id, timestamp, camera_id) VALUES (?, ?, ?);",
                        (frame_id, ts, 'edge')  # Sử dụng camera_id mặc định
                    )
                # ### <<< KẾT THÚC SỬA LỖI
                
                await self.db.execute("INSERT INTO Detections(person_id, timestamp, camera_id, frame_id) VALUES (?, ?, ?, ?);", (pid, ts, None, frame_id))

                # ... (Các câu lệnh INSERT vào FeatureBank và FaceIDBank không đổi) ...
                if feat_v is not None:
                    cur = await self.db.execute("INSERT INTO FeatureBank(person_id) VALUES (?) RETURNING feature_id;", (pid,))
                    feat_id = (await cur.fetchone())[0]
                    await self.db.execute("INSERT INTO FeatureBankVec(feature_id, person_id, feature_vec) VALUES (?, ?, vec_f32(?));", (feat_id, pid, feat_v.tobytes()))
                if face_v is not None:
                    cur2 = await self.db.execute("INSERT INTO FaceIDBank(person_id) VALUES (?) RETURNING face_id;", (pid,))
                    face_id = (await cur2.fetchone())[0]
                    await self.db.execute("INSERT INTO FaceIDBankVec(face_id, person_id, face_vec) VALUES (?, ?, vec_f32(?));", (face_id, pid, face_v.tobytes()))

                await self.db.commit()

            if feat_v is not None: self.feature_history[pid].append(feat_v)
            if face_v is not None: self.face_embedding_history[pid].append(face_v)

            # if bbox_data and head_point_3d and frame_id:
            #     self.tracking_manager_3D.add_track(object_id=pid, bbox=bbox_data, point3d=head_point_3d, frame_id=frame_id)
            #     logger.info(f"Added new 3D track for ID: {pid}")
            if bbox_data is not None and frame_id is not None:
                self.tracking_manager.add_track(pid, bbox_data, frame_id, feature=feat_v)
                logger.info("Added new track to tracking_manager")

            return pid
        except Exception as e:
            logger.exception(f"[DB] Failed to create person: {e}")
            return None

    async def update_person(self, person_id, gender=None, race=None, age=None,
                            body_color=None, feature=None, face_embedding=None,
                            est_height_m=None, head_point_3d=None,
                            bbox_data=None, frame_id=None):
        await self._ensure_db_ready()
        try:
            person_id = self.alias_map.get(person_id, person_id)
            feat_v = self.normalize(np.asarray(feature, np.float32) if feature is not None else None)
            face_v = self.normalize(np.asarray(face_embedding, np.float32) if face_embedding is not None else None)
            _ = body_color
            ts = asyncio.get_event_loop().time()

            async with self.db_lock:
                if frame_id is not None:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO Frames(frame_id, timestamp, camera_id) VALUES (?, ?, ?);",
                        (frame_id, ts, 'edge')
                    )
                
                await self.db.execute("INSERT INTO Detections(person_id, timestamp, camera_id, frame_id) VALUES (?, ?, ?, ?);", (person_id, ts, None, frame_id))

                if feat_v is not None: self.feature_history[person_id].append(feat_v)
                if face_v is not None: self.face_embedding_history[person_id].append(face_v)

                # ### <<< THAY ĐỔI: Sử dụng `[person_id]` thay vì `.get(person_id)`
                # Điều này đảm bảo `compute_ema` luôn nhận được một deque (có thể rỗng)
                mv_feat, mv_face = await asyncio.gather(
                    asyncio.to_thread(compute_ema, self.feature_history[person_id]),
                    asyncio.to_thread(compute_ema, self.face_embedding_history[person_id])
                )
                # ### <<< KẾT THÚC THAY ĐỔI

                if mv_feat is not None:
                    await self.db.execute("UPDATE PersonsVec SET feature_mean = vec_f32(?) WHERE person_id = ?;", (mv_feat.tobytes(), person_id))
                if mv_face is not None:
                    await self.db.execute("UPDATE FaceVector SET face_embedding = vec_f32(?) WHERE person_id = ?;", (mv_face.tobytes(), person_id))

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

            # if bbox_data and head_point_3d and frame_id:
            #     self.tracking_manager_3D.update_track(object_id=person_id, bbox=bbox_data, point3d=head_point_3d, frame_id=frame_id)
            #     logger.info(f"Updated 3D track for ID: {person_id}")
            if bbox_data is not None and frame_id is not None:
                self.tracking_manager.update_track(person_id, bbox_data, frame_id, feature=feat_v)
                logger.info("Updated track in tracking_manager")

            return True
        except Exception as e:
            logger.exception(f"[DB] Failed to update_person: {e}")
            return None
        
    async def match_id(self, new_gender, new_race, new_age, new_body_color, new_feature, new_face_embedding=None,
                       new_height_m=None, new_head_point_3d=None,
                       bbox_data=None, frame_id=None):
        """
        Tìm ID khớp nhất cho một phát hiện mới.
        """
        await self._ensure_db_ready()
        try:
            start = time.time()

            if new_feature is None and new_face_embedding is None:
                logger.warning("No feature or face embedding given for matching.")
                return None

            nf = self.normalize(np.asarray(new_feature, dtype=np.float32)) if new_feature is not None else None
            nface = self.normalize(np.asarray(new_face_embedding, dtype=np.float32)) if new_face_embedding is not None else None
            
            _ = new_body_color
            _ = new_height_m

            async with self.db_lock:
                sql_parts = ["1=1"]
                params = []
                if new_age is not None: sql_parts.append("(age = ? OR age IS NULL)"); params.append(new_age)
                if new_gender is not None: sql_parts.append("(gender = ? OR gender IS NULL)"); params.append(new_gender)
                if new_race is not None: sql_parts.append("(race = ? OR race IS NULL)"); params.append(new_race)
                
                query = f"SELECT person_id FROM PersonsMeta WHERE {' AND '.join(sql_parts)}"
                cur = await self.db.execute(query, params)
                candidates = [row[0] for row in await cur.fetchall()]
                
                if not candidates:
                    logger.info("No candidates after metadata filtering.")
                    return None

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
                                logger.info(f"Face match passed for {best_face_pid} (sim={sim_bank:.4f}) in {time.time()-start:.2f}s")
                                return best_face_pid

            # if bbox_data is not None and new_head_point_3d is not None and frame_id is not None:
            #     matched_pid = self.tracking_manager_3D.match(
            #         bbox=bbox_data, point3d=new_head_point_3d, frame_id=frame_id
            #     )
            #     if matched_pid is not None:
            #         logger.info(f"MATCH ID with 3D tracking: {matched_pid}")
            #         return matched_pid
            #     else:
            #         logger.info("NO-MATCH_ID with 3D tracking")
            if bbox_data is not None and frame_id is not None:
                matched_pid = self.tracking_manager.match(bbox_data, frame_id, nf)
                if matched_pid is not None:
                    logger.info(f"MATCH ID with kalman filter: {matched_pid}")
                    return matched_pid
                else:
                    logger.info("NO-MATCH_ID with kalman filter")
            
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
                                logger.info(f"Hard-feature check passed for {best_pid} (sim={sim_bank:.4f})")
                                return best_pid

            return None
        except Exception as e:
            logger.exception(f"[DB] Failed to match_id: {e}")
            return None



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