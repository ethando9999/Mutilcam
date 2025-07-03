import os
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

import numpy as np
import aiosqlite
import uuid
# from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_python_orangepi import get_logger
from utils.pose_color_signature_new import preprocess_color

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
        δ_margin = 0.05
    ):
        self.db_path = db_path
        self.feature_threshold = feature_threshold
        self.color_threshold = color_threshold 
        self.avg_threshold = avg_threshold
        self.top_k = top_k

        # ------------------------------------------------------------------
        # THÊM: trọng số cho thigh & torso trên 45-dim
        # ------------------------------------------------------------------
        self.thigh_weight = thigh_weight  
        self.torso_weight = torso_weight

        # 15 vùng → 15 × 3 = 45 chiều
        # self.sum_color = np.zeros(51, dtype=np.float64)
        # self.count_color = np.zeros(51, dtype=np.int64)
        self.sum_color = np.zeros(45, dtype=np.float64)
        self.count_color = np.zeros(45, dtype=np.int64)
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
        # THÊM: trọng số cho feature & color
        # ------------------------------------------------------------------
        self.feature_weight = feature_weight
        self.color_weight = color_weight    
        assert abs(self.feature_weight + self.color_weight - 1.0) < 1e-6, "Weights must sum to 1"

        # history for moving average per person
        self.feature_history = defaultdict(lambda: deque(maxlen=10))
        self.color_history   = defaultdict(lambda: deque(maxlen=10))

        # initialization
        self.db_lock = asyncio.Lock()
        self.db_init = asyncio.Event()
        self.db = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vec_ext = '/usr/local/lib/vec0.so'
        self.connection = None
        self.channel = None
        self.alias_map: dict[str, str] = {} 

        self.δ0_feat = δ0_feat
        self.δ_margin = δ_margin

        # start init tasks
        asyncio.create_task(self._init_database())
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
            await db.load_extension(self.vec_ext)

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
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # def preprocess_color(self, body_color: np.ndarray):
    #     """
    #     - body_color: ndarray shape (15,3)  # 15 vùng (12 COCO + 2 thigh + 1 torso)
    #     - flatten → shape (45,)
    #     - NaN → thay bằng global_color_mean
    #     - Áp dụng trọng số cho thigh (vùng 12,13), torso (vùng 14)
    #     - normalize
    #     - Trả về: (vector đã chuẩn hóa, mask)
    #     """
    #     if body_color.shape != (15, 3):
    #         raise ValueError("body_color phải có shape (15, 3)")
        
    #     flat = body_color.flatten().astype(np.float32)       # (45,)
    #     mask = (~np.isnan(flat)).astype(np.float32)          # (45,)

    #     filled = np.where(np.isnan(flat), self.global_color_mean, flat)

    #     weights = np.ones(45, dtype=np.float32)

    #     # Thigh1 = region 12 → flat[36:39]
    #     weights[36:39] *= self.thigh_weight
    #     # Thigh2 = region 13 → flat[39:42]
    #     weights[39:42] *= self.thigh_weight
    #     # Torso  = region 14 → flat[42:45]
    #     weights[42:45] *= self.torso_weight

    #     weighted = filled * weights
    #     normed = self.normalize(weighted)

    #     return normed, mask
    def preprocess_color(self, body_color: np.ndarray):
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


    async def create_person(self, gender, race, age, body_color, feature):  
        await self._ensure_db_ready()
        try:
            # 1) Chuẩn bị feat_v (normalized) và color_v (preprocessed) nếu có
            feat_v = None
            if feature is not None:
                arr = np.asarray(feature, dtype=np.float32)
                if arr.shape != (512,):
                    raise ValueError("feature phải là 512-dim")
                feat_v = self.normalize(arr)

            color_v = None
            if body_color is not None:
                bc = np.asarray(body_color, dtype=np.float32)
                color_v, _ = self.preprocess_color(bc)

            pid = str(uuid.uuid4())

            async with self.db_lock:
                # 2) Chèn vào bảng Persons
                await self.db.execute(
                    "INSERT INTO Persons(person_id) VALUES (?);",
                    (pid,)
                )

                # 3) Chèn embedding vào PersonsVec (có feature + body_color)
                await self.db.execute(
                    """
                    INSERT INTO PersonsVec(
                        person_id,
                        feature_mean,
                        body_color_mean
                    ) VALUES (
                        ?,
                        vec_f32(?),
                        vec_f32(?)
                    );
                    """,
                    (
                        pid,
                        feat_v.tobytes() if feat_v is not None else None,
                        color_v.tobytes() if color_v is not None else None
                    )
                )

                # 4) Chèn metadata vào PersonsMeta
                await self.db.execute(
                    "INSERT INTO PersonsMeta(person_id, age, gender, race) VALUES (?, ?, ?, ?);",
                    (pid, age, gender, race)
                )

                # 5) Chèn detection khởi tạo vào Detections
                ts = asyncio.get_event_loop().time()
                await self.db.execute(
                    """
                    INSERT INTO Detections(
                        person_id,
                        timestamp,
                        camera_id,
                        frame_id
                    ) VALUES (?, ?, ?, ?);
                    """,
                    (pid, ts, 'edge', 'frame0')
                )

                # 6) Chèn vào PersonsCenter (chỉ feature):
                #    centroid_f = feat_v, radius_f = δ0_feat, n_feat = 1
                if feat_v is not None:
                    await self.db.execute(
                        """
                        INSERT INTO PersonsCenter(
                            person_id,
                            centroid_f,
                            radius_f,
                            n_feat
                        ) VALUES (
                            ?,
                            vec_f32(?),
                            ?,
                            ?
                        );
                        """,
                        (
                            pid,
                            feat_v.tobytes(),
                            self.δ0_feat,
                            1
                        )
                    )
                else:
                    # Nếu không có feature (rất hiếm), chèn NULL, radius=0, n_feat=0
                    await self.db.execute(
                        """
                        INSERT INTO PersonsCenter(
                            person_id,
                            centroid_f,
                            radius_f,
                            n_feat
                        ) VALUES (
                            ?, NULL, 0.0, 0
                        );
                        """,
                        (pid,)
                    )

                # 7) Commit một lần duy nhất trong transaction
                await self.db.commit()

            # 8) Checkpoint WAL
            await self.db.execute("PRAGMA wal_checkpoint(FULL);")
            await self.db.commit()

            # 9) Cập nhật RAM history (nếu dùng)
            if feat_v is not None:
                self.feature_history[pid].append(feat_v)
            if color_v is not None:
                self.color_history[pid].append(color_v)
                # self._update_global_color_stats(color_v)

            return pid

        except Exception as e:
            logger.exception(f"[DB] Failed to create_person: {e}")
            return None

    async def update_person(self, person_id, gender, race, age, body_color, feature):
        await self._ensure_db_ready()
        try:
            # 0) Remap nếu person_id là temp_id đã bị replace
            person_id = self.alias_map.get(person_id, person_id)

            # 1) Tính feat_v và color_v (riêng phần CBS chỉ cần feat_v)
            feat_v = None
            if feature is not None:
                arr = np.asarray(feature, dtype=np.float32)
                if arr.shape != (512,):
                    raise ValueError("feature phải là 512-dim")
                feat_v = self.normalize(arr)

            color_v = None
            if body_color is not None:
                bc = np.asarray(body_color, dtype=np.float32)
                color_v, _ = self.preprocess_color(bc)

            async with self.db_lock:
                ts = asyncio.get_event_loop().time()

                # --- 2) Ghi detection (luôn luôn muốn làm ngay)
                await self.db.execute(
                    """
                    INSERT INTO Detections(person_id, timestamp, camera_id, frame_id)
                    VALUES (?, ?, ?, ?);
                    """,
                    (person_id, ts, 'edge', 'frame0')
                )

                # --- 3) Cập nhật RAM history (nếu đang dùng để compute moving-average)
                if feat_v is not None:
                    self.feature_history[person_id].append(feat_v)
                if color_v is not None:
                    self.color_history[person_id].append(color_v)
                    # self._update_global_color_stats(color_v)

                # --- 4) Tính moving-average cho PersonsVec (nếu cần)
                mv_feat = (
                    np.mean(self.feature_history[person_id], axis=0)
                    if self.feature_history[person_id] else None
                )
                mv_color = (
                    np.mean(self.color_history[person_id], axis=0)
                    if self.color_history[person_id] else None
                )

                # Chuẩn bị các câu UPDATE; chỉ thực hiện nếu có mv_feat hoặc mv_color
                if mv_feat is not None or mv_color is not None:
                    await self.db.execute(
                        """
                        UPDATE PersonsVec
                        SET feature_mean    = vec_f32(?),
                            body_color_mean = vec_f32(?)
                        WHERE person_id = ?;
                        """,
                        (
                            mv_feat.tobytes() if mv_feat is not None else None,
                            mv_color.tobytes() if mv_color is not None else None,
                            person_id
                        )
                    )

                # --- 5) Lấy record cũ từ PersonsCenter (chỉ chứa feature‐center)
                #
                #    Bảng PersonsCenter có schema:
                #      person_id   TEXT PRIMARY KEY,
                #      centroid_f  BLOB,    -- vec_f32(...) của feature centroid
                #      radius_f    REAL,    -- ngưỡng (1 - cosine) cho feature
                #      n_feat      INTEGER  -- số sample đã dùng để build centroid
                #
                row_center = await self.db.execute(
                    """
                    SELECT centroid_f, radius_f, n_feat
                    FROM PersonsCenter
                    WHERE person_id = ?;
                    """,
                    (person_id,)
                )
                rec_center = await row_center.fetchone()
                if rec_center is not None:
                    old_centroid_f_buf, radius_f_old, n_feat_old = rec_center

                    # --- 5.1) Nếu có feature mới và centroid cũ tồn tại → cập nhật centroid & radius
                    if feat_v is not None and old_centroid_f_buf is not None:
                        old_centroid_f = np.frombuffer(old_centroid_f_buf, dtype=np.float32)
                        n_feat_old = int(n_feat_old)

                        # 1) incremental mean raw
                        centroid_raw = (old_centroid_f * n_feat_old + feat_v) / (n_feat_old + 1)
                        # 2) normalize centroid
                        norm = np.linalg.norm(centroid_raw)
                        centroid_f_new = centroid_raw / norm if norm > 0 else centroid_raw
                        n_feat_new = n_feat_old + 1

                        # 3) cosine distance = 1 - cosine_similarity
                        dist_f = 1.0 - float(np.dot(feat_v, centroid_f_new))
                        # 4) update radius với margin
                        radius_f_new = max(radius_f_old, dist_f) + self.δ_margin

                    else:
                        # lần đầu hoặc không có feature mới
                        if feat_v is not None:
                            centroid_f_new = feat_v.copy()   # đã normalize đầu vào
                            radius_f_new   = 0.0 + self.δ_margin
                            n_feat_new     = 1
                        else:
                            centroid_f_new = None
                            radius_f_new   = float(radius_f_old)
                            n_feat_new     = int(n_feat_old)

                    # --- 5.2) Thực hiện UPDATE vào PersonsCenter (dùng COALESCE để giữ nguyên nếu không có centroid mới)
                    await self.db.execute(
                        """
                        UPDATE PersonsCenter
                        SET
                            centroid_f = COALESCE(vec_f32(?), centroid_f),
                            radius_f   = ?,
                            n_feat     = ?
                        WHERE person_id = ?;
                        """,
                        (
                            centroid_f_new.tobytes() if centroid_f_new is not None else None,
                            radius_f_new,
                            n_feat_new,
                            person_id
                        )
                    )

                # --- 6) Cập nhật Metadata (nếu cần)
                fields, params = [], []
                if age is not None:
                    fields.append("age = ?");    params.append(age)
                if gender is not None:
                    fields.append("gender = ?"); params.append(gender)
                if race is not None:
                    fields.append("race = ?");   params.append(race)

                if fields:
                    sql_meta = f"UPDATE PersonsMeta SET {', '.join(fields)} WHERE person_id = ?;"
                    params.append(person_id)
                    await self.db.execute(sql_meta, params)

                # --- 7) Cuối cùng: commit và checkpoint một lần duy nhất
                await self.db.commit()
                await self.db.execute("PRAGMA wal_checkpoint(FULL);")
                await self.db.commit()

        except Exception as e:
            logger.exception(f"[DB] Failed to update_person: {e}")
            return None

    async def match_id(
        self,
        new_gender: str | None,
        new_race: str | None,
        new_age: int | None,
        new_body_color: np.ndarray | None,
        new_feature: np.ndarray | None,
    ):
        """
        Stage 1: Metadata filter
        Stage 2: Combined vector filter (feature & color) + lọc threshold → shortlist
        Stage 3: Center-Based Scoring dựa trên feature centroid & radius
        """
        await self._ensure_db_ready()

        if new_feature is None:
            logger.warning("Không có feature để match.")
            return None

        nf = self.normalize(np.asarray(new_feature, dtype=np.float32))
        if nf.shape != (512,):
            raise ValueError("new_feature phải là vector 512-D")

        nc = None
        if new_body_color is not None:
            bc = np.asarray(new_body_color, dtype=np.float32)
            nc, _ = self.preprocess_color(bc)

        start = time.time()

        rows_cb = []
        async with self.db_lock:
            # 1. Metadata filter
            query_meta = """
                SELECT person_id
                FROM PersonsMeta
                WHERE ((?1 IS NULL) OR (age    = ?1 OR age    IS NULL))
                AND ((?2 IS NULL) OR (gender = ?2 OR gender IS NULL))
                AND ((?3 IS NULL) OR (race   = ?3 OR race   IS NULL));
            """
            cur_meta = await self.db.execute(query_meta, (new_age, new_gender, new_race))
            candidates = [row[0] for row in await cur_meta.fetchall()]
            if not candidates:
                logger.warning("FACE khác -> tạo id mới!")
                return None

            placeholders = ",".join("?" for _ in candidates)
            k = min(self.top_k, len(candidates))

            # 2. Vector matching (feature + color)
            if nc is not None:
                sql_cb = f"""
                    SELECT f.person_id, f.distance AS feature_distance, c.distance AS color_distance
                    FROM (
                        SELECT person_id, distance
                        FROM PersonsVec
                        WHERE person_id IN ({placeholders})
                        AND feature_mean MATCH vec_f32(?)
                        AND k = ?
                    ) AS f
                    JOIN (
                        SELECT person_id, distance
                        FROM PersonsVec
                        WHERE person_id IN ({placeholders})
                        AND body_color_mean MATCH vec_f32(?)
                        AND k = ?
                    ) AS c ON f.person_id = c.person_id;
                """
                params_cb = (
                    candidates + [nf.tobytes(), k] +
                    candidates + [nc.tobytes(), k]
                )
                cur_cb = await self.db.execute(sql_cb, params_cb)
                raw_rows = await cur_cb.fetchall()
            else:
                sql_cb = f"""
                    SELECT person_id, distance AS feature_distance
                    FROM PersonsVec
                    WHERE person_id IN ({placeholders})
                    AND feature_mean MATCH vec_f32(?)
                    AND k = ?
                """
                params_cb = candidates + [nf.tobytes(), k]
                cur_cb = await self.db.execute(sql_cb, params_cb)
                raw_rows = [(pid, dist, None) for pid, dist in await cur_cb.fetchall()]

            # 2.1 Lọc theo threshold, phân biệt lý do loại bỏ
            for pid, feat_dist, color_dist in raw_rows:
                feat_sim = 1.0 - feat_dist
                color_sim = 1.0 - color_dist if color_dist is not None else 1.0

                if feat_sim < self.feature_threshold:
                    logger.warning(f"PID {pid} bị loại do FEATURE similarity thấp: {feat_sim:.3f} < {self.feature_threshold}")
                    continue

                if color_dist is not None and color_sim < self.color_threshold:
                    logger.warning(f"PID {pid} bị loại do COLOR similarity thấp: {color_sim:.3f} < {self.color_threshold}")
                    continue

                rows_cb.append((pid, feat_dist, color_dist))

            if not rows_cb:
                logger.warning("Không có ai vượt qua ngưỡng FEATURE và COLOR → tạo ID mới!")
                return None
            
        # best_pid, best_dist, color_dist = min(rows_cb, key=lambda t: t[1])
        # best_sim = 1.0 - best_dist
        # logger.info(f"MATCH ID {best_pid} with sim={best_sim:.4f}")
        # return best_pid

            # 3. Query PersonsCenter cho những người còn lại
            pids_cb = [pid for pid, _, _ in rows_cb]
            placeholders2 = ",".join("?" for _ in pids_cb)
            sql_center = f"""
                SELECT person_id, centroid_f, radius_f
                FROM PersonsCenter
                WHERE person_id IN ({placeholders2});
            """
            cur_center = await self.db.execute(sql_center, pids_cb)
            center_rows = await cur_center.fetchall()


        # 4. Center-Based Scoring
        center_feat = {
            pid: (np.frombuffer(cf_buf, dtype=np.float32), float(rad_f))
            for pid, cf_buf, rad_f in center_rows
        }

        best_pid = None
        best_dist = float("inf")
        for pid, _, _ in rows_cb:
            if pid not in center_feat:
                continue
            centroid_f, radius_f = center_feat[pid]
            dist_f = 1.0 - float(np.dot(nf, centroid_f))
            if dist_f <= radius_f and dist_f < best_dist:
                best_dist = dist_f
                best_pid = pid

        if best_pid is not None:
            logger.debug(
                "MATCH pid=%s qua CBS-feature (dist=%.4f) trong %.2fs",
                best_pid,
                best_dist,
                time.time() - start,
            )
            return best_pid
        logger.warning("Center-Based không khớp -> tạo id mới!")

        return None


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

    async def find_closest_by_id(
        self, temp_id: str
    ) -> tuple[str | None, float]:
        """
        Chỉ dùng DB để tìm person_id gần nhất (k=1) với temp_id theo feature_mean.
        Trả về (best_id, best_similarity) hoặc (None, 0.0) nếu không tìm được.
        """
        await self._ensure_db_ready()

        sql = """
        WITH query_vec AS (
        SELECT feature_mean AS buf
        FROM PersonsVec
        WHERE person_id = ?
        )
        SELECT p.person_id,
            1.0 - p.distance AS similarity
        FROM PersonsVec AS p, query_vec
        WHERE p.person_id != ?
        AND p.feature_mean MATCH vec_f32(query_vec.buf)
        AND p.k = 1
        LIMIT 1;
        """
        try:
            async with self.db.execute(sql, [temp_id, temp_id]) as cur:
                row = await cur.fetchone()
                if row:
                    pid, sim = row
                    return pid, max(sim, 0.0)
        except Exception as e:
            logger.error(f"[find_closest_by_id] SQL error: {e}")

        return None, 0.0


