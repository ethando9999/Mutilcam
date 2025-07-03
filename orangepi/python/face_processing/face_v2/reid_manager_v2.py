import asyncio
import aiosqlite
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_python_orangepi import get_logger
from concurrent.futures import ThreadPoolExecutor
import aio_pika
import json
from utils.create_id_db import initialize_db
import time
from collections import defaultdict, deque
import uuid
import random
from .config import get_config
import datetime

logger = get_logger(__name__)

class ReIDManager:
    def __init__(self, temp_db_path='temp.db', main_db_path='database.db', feature_threshold=0.65,  
                 remote_feature_threshold=0.7, color_threshold=0.5, face_threshold=0.7, face_id_merge=0.8, 
                 max_items=10, rabbitmq_url="amqp://new_user_rpi:123456@192.168.1.15/", top_k=5, 
                 global_color_mean=None, device_id="rpi", request_timeout=3, wal_checkpoint_interval=300, 
                 sync_frame_limit=10):
        config = get_config()
        
        # Cấu hình cơ bản
        self.temp_db_path = temp_db_path
        self.main_db_path = main_db_path
        self.feature_threshold = config.get('feature_threshold', feature_threshold)
        self.remote_feature_threshold = config.get('remote_feature_threshold', remote_feature_threshold)
        self.color_threshold = config.get('color_threshold', color_threshold)
        self.face_threshold = config.get('face_threshold', face_threshold)
        self.face_id_merge = config.get('face_id_merge', face_id_merge)
        self.max_items = max_items
        self.top_k = top_k
        self.device_id = config.get('device_id', device_id)
        self.request_timeout = config.get('request_timeout', request_timeout)
        self.wal_checkpoint_interval = config.get('wal_checkpoint_interval', wal_checkpoint_interval)
        self.sync_frame_limit = config.get('sync_frame_limit', sync_frame_limit)
        
        # Trọng số tối ưu hóa
        self.face_weight = config.get('face_weight', 0.7)
        self.color_weight = config.get('color_weight', 0.1)
        self.feature_weight = config.get('feature_weight', 0.2)
        self.combined_threshold = config.get('combined_threshold', 0.6)
        
        # Khởi tạo các thành phần
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.connection = None
        self.channel = None
        self.rabbitmq_url = config.get('rabbitmq_url', rabbitmq_url)
        self.vec_extension_path = '/usr/local/lib/vec0.so'
        self.db_lock = asyncio.Lock()
        self.use_vec = False
        self.temp_db = None
        self.main_db = None
        self.rare_color = np.array([255, 0, 255], dtype=np.float32)
        self.pending_requests = {}
        self.update_counts = defaultdict(int)
        self.first_seen = defaultdict(float)
        self.person_buffer = {}
        self.recently_synced = defaultdict(int)

        # Thống kê màu sắc toàn cục
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

        # Lịch sử đặc trưng
        self.feature_history = defaultdict(lambda: deque(maxlen=max_items))
        self.color_history = defaultdict(lambda: deque(maxlen=max_items))
        self.face_embedding_history = defaultdict(lambda: deque(maxlen=max_items))

        self._rabbitmq_ready = asyncio.Event()
        asyncio.create_task(self._start_background_tasks())

    ### Các phương thức hỗ trợ
    def normalize(self, v: np.ndarray) -> np.ndarray:
        """Chuẩn hóa vector."""
        if v is None or v.size == 0:
            return v
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def preprocess_color(self, body_color: np.ndarray):
        """Chuẩn bị dữ liệu màu sắc."""
        if body_color is None:
            return None, None
        flat = body_color.flatten().astype(np.float32)
        mask = (~np.isnan(flat)).astype(np.float32)
        filled = np.where(np.isnan(flat), self.global_color_mean, flat)
        normed = self.normalize(filled)
        return normed, mask

    def _update_global_color_stats(self, color_mean: np.ndarray):
        """Cập nhật thống kê màu sắc toàn cục."""
        if color_mean is not None and color_mean.ndim == 1:
            valid = ~np.isnan(color_mean)
            self.sum_color[valid] += color_mean[valid]
            self.count_color[valid] += 1
            self.global_color_mean[valid] = (
                self.sum_color[valid] / self.count_color[valid]
            ).astype(np.float32)

    def _ensure_scalar(self, value):
        """Đảm bảo giá trị là vô hướng."""
        if isinstance(value, tuple):
            return value[0] if value else None
        return value

    async def _get_history_or_db(self, pid, history_attr, table, column):
        """Lấy dữ liệu từ lịch sử hoặc cơ sở dữ liệu."""
        history = getattr(self, history_attr)
        if pid in history and history[pid]:
            feats = list(history[pid])
            if not all(feat.ndim == 1 for feat in feats):
                logger.error(f"[{self.device_id}] Lịch sử đặc trưng không phải là mảng 1 chiều cho {pid}")
                return None
            return np.mean(feats, axis=0)
        else:
            async with self.temp_db.execute(f"SELECT {column} FROM {table} WHERE person_id = ?", (pid,)) as cursor:
                row = await cursor.fetchone()
            if row and row[0]:
                vec = np.frombuffer(row[0], dtype=np.float32)
                if vec.ndim != 1:
                    logger.error(f"[{self.device_id}] Dữ liệu {column} từ DB cho {pid} không phải mảng 1 chiều")
                    return None
                return vec
            return None

    ### Khởi tạo và quản lý nền tảng
    async def _start_background_tasks(self):
        await self.setup_rabbitmq()
        self._rabbitmq_ready.set()
        await asyncio.gather(
            self.initialize_databases(),
            self.maintain_rabbitmq_connection(),
            self.consume_id_results(),
            self.consume_remote_requests(),
            self.consume_responses(),
            self.periodic_move_to_main_db(),
            self.periodic_wal_checkpoint(),
            self.schedule_daily_clear()
        )

    async def initialize_databases(self):
        async with self.db_lock:
            retries = 3
            for attempt in range(retries):
                try:
                    self.use_vec = await initialize_db(db_path=self.temp_db_path, vec_extension_path=self.vec_extension_path)
                    await initialize_db(db_path=self.main_db_path, vec_extension_path=self.vec_extension_path)
                    self.temp_db = await aiosqlite.connect(self.temp_db_path)
                    self.main_db = await aiosqlite.connect(self.main_db_path)

                    for db in [self.temp_db, self.main_db]:
                        if self.use_vec:
                            await db.enable_load_extension(True)
                            await db.load_extension(self.vec_extension_path)
                        await db.execute('CREATE TABLE IF NOT EXISTS PersonSources (person_id TEXT PRIMARY KEY, source TEXT, sync_time REAL)')
                        await db.execute('CREATE TABLE IF NOT EXISTS FaceVector (person_id TEXT PRIMARY KEY, face_embedding BLOB)')
                        await db.execute('CREATE TABLE IF NOT EXISTS FaceVectorHistory (person_id TEXT, face_embedding BLOB, timestamp REAL, PRIMARY KEY (person_id, timestamp))')
                        await db.execute('CREATE TABLE IF NOT EXISTS Persons (person_id TEXT PRIMARY KEY, frame BLOB)')
                        await db.execute('CREATE TABLE IF NOT EXISTS Detections (person_id TEXT, timestamp REAL, camera_id TEXT, frame_id TEXT, device_id TEXT)')
                        await db.execute('CREATE TABLE IF NOT EXISTS Face (person_id TEXT, gender TEXT, race TEXT, age TEXT)')
                        await db.execute('CREATE INDEX IF NOT EXISTS idx_persons_person_id ON Persons(person_id)')
                        await db.execute('CREATE INDEX IF NOT EXISTS idx_detections_person_id ON Detections(person_id)')
                        await db.execute('CREATE INDEX IF NOT EXISTS idx_face_person_id ON Face(person_id)')
                        await db.execute('CREATE INDEX IF NOT EXISTS idx_face_gender_race_age ON Face(gender, race, age)')
                        await db.commit()
                    logger.info(f"[{self.device_id}] Sử dụng sqlite-vec: {self.use_vec}")
                    break
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi khởi tạo cơ sở dữ liệu (lần {attempt + 1}/{retries}): {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise
        logger.info(f"[{self.device_id}] TempDB và MainDB đã được khởi tạo.")

    async def setup_rabbitmq(self):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if self.connection and not self.connection.is_closed:
                    await self.connection.close()
                self.connection = await aio_pika.connect_robust(
                    self.rabbitmq_url, heartbeat=120, timeout=30
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=5)

                self.id_results_exchange = await self.channel.declare_exchange(
                    'id_results_exchange', aio_pika.ExchangeType.FANOUT, durable=True
                )
                self.remote_requests_exchange = await self.channel.declare_exchange(
                    'remote_requests_exchange', aio_pika.ExchangeType.FANOUT, durable=True
                )

                id_results_queue_name = f'id_results_{self.device_id}'
                self.id_results_queue = await self.channel.declare_queue(id_results_queue_name, durable=True)
                await self.id_results_queue.bind(self.id_results_exchange)

                remote_requests_queue_name = f'remote_requests_{self.device_id}'
                self.remote_requests_queue = await self.channel.declare_queue(remote_requests_queue_name, durable=True)
                await self.remote_requests_queue.bind(self.remote_requests_exchange)

                response_queue_name = f'remote_response_{self.device_id}_{uuid.uuid4()}'
                self.response_queue = await self.channel.declare_queue(
                    response_queue_name, durable=False, exclusive=True, auto_delete=True
                )
                logger.info(f"[{self.device_id}] Đã khai báo hàng đợi phản hồi: {response_queue_name}")
                logger.info(f"[{self.device_id}] Kết nối RabbitMQ thành công")
                break
            except Exception as e:
                logger.error(f"[{self.device_id}] Lỗi kết nối RabbitMQ (lần {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt + random.uniform(0, 1), 30)
                    await asyncio.sleep(delay)
                else:
                    raise

    async def maintain_rabbitmq_connection(self):
        while True:
            try:
                if self.connection is None or self.connection.is_closed or self.channel is None or self.channel.is_closed:
                    logger.warning(f"[{self.device_id}] Mất kết nối RabbitMQ, đang kết nối lại...")
                    await self.setup_rabbitmq()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"[{self.device_id}] Lỗi duy trì kết nối RabbitMQ: {str(e)}")
                await asyncio.sleep(5)

    async def periodic_wal_checkpoint(self):
        while True:
            await asyncio.sleep(self.wal_checkpoint_interval)
            async with self.db_lock:
                try:
                    for db_name, db in [("TempDB", self.temp_db), ("MainDB", self.main_db)]:
                        if db:
                            await db.execute("PRAGMA wal_checkpoint(FULL);")
                            await db.commit()
                            logger.info(f"[{self.device_id}] Đã thực hiện WAL checkpoint trên {db_name}")
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi trong WAL checkpoint: {str(e)}")

    async def clear_person_vector(self):
        async with self.db_lock:
            await self.temp_db.execute("DELETE FROM PersonVector")
            await self.temp_db.commit()
            self.feature_history.clear()
            self.color_history.clear()
            self.face_embedding_history.clear()
            logger.info(f"[{self.device_id}] Đã xóa bảng PersonVector tại {datetime.datetime.now()}")

    async def schedule_daily_clear(self):
        while True:
            now = datetime.datetime.now()
            next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)
            if next_run < now:
                next_run += datetime.timedelta(days=1)
            wait_seconds = (next_run - now).total_seconds()
            logger.info(f"[{self.device_id}] Xóa PersonVector tiếp theo sau {wait_seconds} giây tại {next_run}")
            await asyncio.sleep(wait_seconds)
            await self.clear_person_vector()

    ### Xử lý thông điệp RabbitMQ
    async def publish_id(self, person_id, gender, race, age, body_color_mean, feature_mean, first_seen, face_embedding):
        gender = self._ensure_scalar(gender)
        race = self._ensure_scalar(race)
        age = self._ensure_scalar(age)
        await self._ensure_rabbitmq_ready()
        if not self.channel or self.channel.is_closed:
            await self.setup_rabbitmq()
        try:
            message_body = json.dumps({
                "type": "person",
                "person_id": person_id,
                "gender": gender,
                "race": race,
                "age": age,
                "body_color_mean": body_color_mean.tolist() if body_color_mean is not None else None,
                "feature_mean": feature_mean.tolist() if feature_mean is not None else None,
                "face_embedding": face_embedding.tolist() if face_embedding is not None else None,
                "device_id": self.device_id,
                "timestamp": asyncio.get_event_loop().time(),
                "first_seen": first_seen
            }).encode()
            message = aio_pika.Message(body=message_body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT)
            await self.id_results_exchange.publish(message, routing_key='')
            logger.info(f"[{self.device_id}] Đã publish dữ liệu ReID cho person_id {person_id}")
        except Exception as e:
            logger.error(f"[{self.device_id}] Lỗi khi publish dữ liệu: {str(e)}")
            await self.setup_rabbitmq()

    async def publish_merge(self, from_id, to_id):
        await self._ensure_rabbitmq_ready()
        if not self.channel or self.channel.is_closed:
            await self.setup_rabbitmq()
        try:
            message_body = json.dumps({
                "type": "merge",
                "from_id": from_id,
                "to_id": to_id,
                "device_id": self.device_id,
                "timestamp": asyncio.get_event_loop().time()
            }).encode()
            message = aio_pika.Message(body=message_body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT)
            await self.id_results_exchange.publish(message, routing_key='')
            logger.info(f"[{self.device_id}] Đã publish merge từ {from_id} sang {to_id}")
        except Exception as e:
            logger.error(f"[{self.device_id}] Lỗi khi publish merge: {str(e)}")
            await self.setup_rabbitmq()

    async def consume_id_results(self):
        await self._rabbitmq_ready.wait()
        while True:
            try:
                if not self.channel or self.channel.is_closed or not self.id_results_queue:
                    logger.warning(f"[{self.device_id}] Consumer id_results chưa sẵn sàng, đang kết nối lại...")
                    await self.setup_rabbitmq()
                logger.info(f"[{self.device_id}] Bắt đầu tiêu thụ id_results")
                async with self.id_results_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            try:
                                data = json.loads(message.body.decode())
                                device_id = data.get("device_id")
                                if device_id == self.device_id or device_id is None:
                                    continue

                                if data["type"] == "person":
                                    person_id = data["person_id"]
                                    gender = self._ensure_scalar(data["gender"])
                                    race = self._ensure_scalar(data["race"])
                                    age = self._ensure_scalar(data["age"])
                                    body_color = np.array(data["body_color_mean"], dtype=np.float32) if data.get("body_color_mean") else None
                                    feature = np.array(data["feature_mean"], dtype=np.float32) if data.get("feature_mean") else None
                                    face_embedding = np.array(data["face_embedding"], dtype=np.float32) if data.get("face_embedding") else None
                                    first_seen = data.get("first_seen", asyncio.get_event_loop().time())

                                    logger.info(f"[{self.device_id}] Nhận id_results cho person_id {person_id} từ {device_id}")
                                    async with self.db_lock:
                                        async with self.temp_db.execute('SELECT person_id FROM Persons WHERE person_id = ?', (person_id,)) as cur:
                                            exists = await cur.fetchone()

                                    if exists:
                                        await self.update_person(person_id, gender, race, age, body_color, feature, face_embedding)
                                    else:
                                        matched_id, sim = await self.match_id_synced(gender, race, age, body_color, feature, face_embedding)
                                        if matched_id and sim >= self.remote_feature_threshold and self.update_counts[matched_id] >= 5:
                                            existing_first_seen = self.first_seen.get(matched_id, float('inf'))
                                            if first_seen < existing_first_seen:
                                                await self.update_person(matched_id, gender, race, age, body_color, feature, face_embedding)
                                                await self.publish_merge(person_id, matched_id)
                                                await self.merge_person_ids(person_id, matched_id)
                                                logger.info(f"[{self.device_id}] Sync: Đã merge {person_id} vào {matched_id}")
                                            else:
                                                self.first_seen[person_id] = first_seen
                                                await self.create_person_in_temp(person_id, gender, race, age, body_color, feature, face_embedding, source='synced')
                                                self.recently_synced[person_id] = 0
                                                await self.publish_merge(matched_id, person_id)
                                                await self.merge_person_ids(matched_id, person_id)
                                                logger.info(f"[{self.device_id}] Sync: Đã merge {matched_id} vào {person_id}")
                                        else:
                                            self.first_seen[person_id] = first_seen
                                            await self.create_person_in_temp(person_id, gender, race, age, body_color, feature, face_embedding, source='synced')
                                            self.recently_synced[person_id] = 0
                                            logger.info(f"[{self.device_id}] Sync: Đã thêm {person_id} vào TempDB")

                                elif data["type"] == "merge":
                                    from_id = data["from_id"]
                                    to_id = data["to_id"]
                                    await self.merge_person_ids(from_id, to_id)
                                    logger.info(f"[{self.device_id}] Đã xử lý merge: từ {from_id} sang {to_id}")
                            except Exception as e:
                                logger.error(f"[{self.device_id}] Lỗi xử lý id_results: {str(e)}")
            except Exception as e:
                logger.error(f"[{self.device_id}] Lỗi trong consumer id_results: {str(e)}")
                await asyncio.sleep(5)

    async def merge_person_ids(self, from_id, to_id):
        async with self.db_lock:
            try:
                await self.temp_db.execute('BEGIN TRANSACTION')
                async with self.temp_db.execute('SELECT person_id FROM Persons WHERE person_id = ?', (to_id,)) as cursor:
                    exists = await cursor.fetchone()
                if not exists:
                    logger.warning(f"[{self.device_id}] to_id {to_id} không tồn tại trong TempDB, không thể merge")
                    await self.temp_db.rollback()
                    return

                await self.temp_db.execute('UPDATE Detections SET person_id = ? WHERE person_id = ?', (to_id, from_id))
                await self.temp_db.execute('UPDATE Face SET person_id = ? WHERE person_id = ?', (to_id, from_id))
                await self.temp_db.execute('UPDATE PersonSources SET person_id = ? WHERE person_id = ?', (to_id, from_id))
                if self.use_vec:
                    await self.temp_db.execute('UPDATE PersonVector SET person_id = ? WHERE person_id = ?', (to_id, from_id))
                    await self.temp_db.execute('UPDATE FaceVector SET person_id = ? WHERE person_id = ?', (to_id, from_id))
                await self.temp_db.execute('DELETE FROM Persons WHERE person_id = ?', (from_id,))
                await self.temp_db.commit()

                if from_id in self.feature_history:
                    self.feature_history[to_id].extend(self.feature_history[from_id])
                    del self.feature_history[from_id]
                if from_id in self.color_history:
                    self.color_history[to_id].extend(self.color_history[from_id])
                    del self.color_history[from_id]
                if from_id in self.face_embedding_history:
                    self.face_embedding_history[to_id].extend(self.face_embedding_history[from_id])
                    del self.face_embedding_history[from_id]
                if from_id in self.first_seen:
                    self.first_seen[to_id] = min(self.first_seen[to_id], self.first_seen[from_id])
                    del self.first_seen[from_id]
                if from_id in self.recently_synced:
                    self.recently_synced[to_id] = min(self.recently_synced.get(to_id, 0), self.recently_synced[from_id])
                    del self.recently_synced[from_id]

                logger.info(f"[{self.device_id}] Đã merge {from_id} vào {to_id} trong TempDB")
            except Exception as e:
                await self.temp_db.rollback()
                logger.error(f"[{self.device_id}] Lỗi khi merge person_ids: {str(e)}")

    async def consume_responses(self):
        await self._rabbitmq_ready.wait()
        while True:
            try:
                if not self.channel or self.channel.is_closed or not self.response_queue:
                    logger.warning(f"[{self.device_id}] Consumer response chưa sẵn sàng, đang kết nối lại...")
                    await self.setup_rabbitmq()
                logger.info(f"[{self.device_id}] Bắt đầu tiêu thụ response từ {self.response_queue.name}")
                async with self.response_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            try:
                                response = json.loads(message.body.decode())
                                request_id = response['request_id']
                                if request_id in self.pending_requests:
                                    sim_scores = self.pending_requests[request_id]
                                    matched_id = response.get('matched_id')
                                    sim_score = response.get('similarity', 0.0)
                                    first_seen = response.get('first_seen', asyncio.get_event_loop().time())
                                    if matched_id:
                                        current_sim, _ = sim_scores.get(matched_id, (0.0, float('inf')))
                                        if sim_score > current_sim:
                                            sim_scores[matched_id] = (sim_score, first_seen)
                                    logger.info(f"[{self.device_id}] Nhận phản hồi cho request_id {request_id} với matched_id={matched_id}")
                            except Exception as e:
                                logger.error(f"[{self.device_id}] Lỗi xử lý response: {str(e)}")
            except Exception as e:
                logger.error(f"[{self.device_id}] Lỗi trong consumer response: {str(e)}")
                await asyncio.sleep(5)

    ### Quản lý person
    async def create_person_in_temp(self, person_id, gender, race, age, body_color, feature, face_embedding, source='local', first_seen=None):
        await self._ensure_db_ready()
        feat_v = self.normalize(np.array(feature, dtype=np.float32)) if feature is not None else None
        color_v, _ = self.preprocess_color(np.array(body_color, dtype=np.float32)) if body_color is not None else (None, None)
        face_v = self.normalize(np.array(face_embedding, dtype=np.float32)) if face_embedding is not None else None
        if first_seen is not None:
            self.first_seen[person_id] = first_seen

        gender = self._ensure_scalar(gender)
        race = self._ensure_scalar(race)
        age = self._ensure_scalar(age)

        # Kiểm tra định dạng dữ liệu
        for name, vec in [("feature", feat_v), ("color", color_v), ("face_embedding", face_v)]:
            if vec is not None and vec.ndim != 1:
                logger.error(f"[{self.device_id}] {name} không phải mảng 1 chiều: {vec.shape}")
                return

        async with self.db_lock:
            retries = 3
            for attempt in range(retries):
                try:
                    await self.temp_db.execute("INSERT OR IGNORE INTO Persons(person_id, frame) VALUES(?, ?)", (person_id, None))
                    if self.use_vec:
                        await self.temp_db.execute(
                            "INSERT OR REPLACE INTO PersonVector(person_id, feature_mean, body_color_mean) VALUES(?, vec_f32(?), vec_f32(?))",
                            (person_id, feat_v.tobytes() if feat_v is not None else None, color_v.tobytes() if color_v is not None else None)
                        )
                        if face_v is not None:
                            await self.temp_db.execute(
                                "INSERT OR REPLACE INTO FaceVector(person_id, face_embedding) VALUES(?, vec_f32(?))",
                                (person_id, face_v.tobytes())
                            )
                            await self.temp_db.execute(
                                "INSERT INTO FaceVectorHistory(person_id, face_embedding, timestamp) VALUES(?, vec_f32(?), ?)",
                                (person_id, face_v.tobytes(), asyncio.get_event_loop().time())
                            )
                    else:
                        await self.temp_db.execute(
                            "UPDATE Persons SET feature_mean=?, body_color_mean=? WHERE person_id=?",
                            (feat_v.tobytes() if feat_v is not None else None, color_v.tobytes() if color_v is not None else None, person_id)
                        )
                    await self.temp_db.execute(
                        "INSERT OR REPLACE INTO Face(person_id, gender, race, age) VALUES (?, ?, ?, ?)",
                        (person_id, gender, race, age)
                    )
                    ts = asyncio.get_event_loop().time()
                    await self.temp_db.execute(
                        "INSERT INTO Detections(person_id, timestamp, camera_id, frame_id, device_id) VALUES (?, ?, ?, ?, ?)",
                        (person_id, ts, self.device_id, 'frame0', self.device_id)
                    )
                    await self.temp_db.execute(
                        "INSERT OR REPLACE INTO PersonSources(person_id, source, sync_time) VALUES(?, ?, ?)",
                        (person_id, source, ts if source == 'synced' else None)
                    )
                    await self.temp_db.commit()
                    logger.info(f"[{self.device_id}] Đã tạo hoặc đồng bộ {person_id} trong TempDB với source {source}")
                    break
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi tạo person trong TempDB (lần {attempt + 1}/{retries}): {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise

        if feat_v is not None:
            self.feature_history[person_id].append(feat_v)
        if color_v is not None:
            self.color_history[person_id].append(color_v)
            self._update_global_color_stats(color_v)
        if face_v is not None:
            self.face_embedding_history[person_id].append(face_v)

    async def create_person(self, gender, race, age, body_color, feature, face_embedding):
        await self._ensure_db_ready()
        feat_v = self.normalize(np.array(feature, dtype=np.float32)) if feature is not None else None
        color_v, _ = self.preprocess_color(np.array(body_color, dtype=np.float32)) if body_color is not None else (None, None)
        face_v = self.normalize(np.array(face_embedding, dtype=np.float32)) if face_embedding is not None else None
        pid = str(uuid.uuid4())
        first_seen = asyncio.get_event_loop().time()
        self.first_seen[pid] = first_seen

        gender = self._ensure_scalar(gender)
        race = self._ensure_scalar(race)
        age = self._ensure_scalar(age)

        # Kiểm tra định dạng dữ liệu
        for name, vec in [("feature", feat_v), ("color", color_v), ("face_embedding", face_v)]:
            if vec is not None and vec.ndim != 1:
                logger.error(f"[{self.device_id}] {name} không phải mảng 1 chiều: {vec.shape}")
                return None

        async with self.db_lock:
            retries = 3
            for attempt in range(retries):
                try:
                    await self.temp_db.execute("INSERT OR IGNORE INTO Persons(person_id, frame) VALUES(?, ?)", (pid, None))
                    async with self.temp_db.execute('SELECT person_id FROM Persons WHERE person_id = ?', (pid,)) as cursor:
                        exists = await cursor.fetchone()
                    if not exists:
                        logger.warning(f"[{self.device_id}] Không thể chèn {pid}, thử ID mới")
                        pid = str(uuid.uuid4())
                        await self.temp_db.execute("INSERT OR IGNORE INTO Persons(person_id, frame) VALUES(?, ?)", (pid, None))
                    if self.use_vec:
                        await self.temp_db.execute(
                            "INSERT OR REPLACE INTO PersonVector(person_id, feature_mean, body_color_mean) VALUES(?, vec_f32(?), vec_f32(?))",
                            (pid, feat_v.tobytes() if feat_v is not None else None, color_v.tobytes() if color_v is not None else None)
                        )
                        if face_v is not None:
                            await self.temp_db.execute(
                                "INSERT OR REPLACE INTO FaceVector(person_id, face_embedding) VALUES(?, vec_f32(?))",
                                (pid, face_v.tobytes())
                            )
                            await self.temp_db.execute(
                                "INSERT INTO FaceVectorHistory(person_id, face_embedding, timestamp) VALUES(?, vec_f32(?), ?)",
                                (pid, face_v.tobytes(), first_seen)
                            )
                    else:
                        await self.temp_db.execute(
                            "UPDATE Persons SET feature_mean=?, body_color_mean=? WHERE person_id=?",
                            (feat_v.tobytes() if feat_v is not None else None, color_v.tobytes() if color_v is not None else None, pid)
                        )
                    await self.temp_db.execute(
                        "INSERT OR REPLACE INTO Face(person_id, gender, race, age) VALUES (?, ?, ?, ?)",
                        (pid, gender, race, age)
                    )
                    ts = asyncio.get_event_loop().time()
                    await self.temp_db.execute(
                        "INSERT INTO Detections(person_id, timestamp, camera_id, frame_id, device_id) VALUES (?, ?, ?, ?, ?)",
                        (pid, ts, self.device_id, f'frame_{int(ts)}', self.device_id)
                    )
                    await self.temp_db.execute(
                        "INSERT OR REPLACE INTO PersonSources(person_id, source, sync_time) VALUES(?, ?, ?)",
                        (pid, 'local', None)
                    )
                    await self.temp_db.commit()
                    logger.info(f"[{self.device_id}] Đã tạo person mới id={pid}")
                    break
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi tạo (lần {attempt + 1}/{retries}): {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise

        if feat_v is not None:
            self.feature_history[pid].append(feat_v)
        if color_v is not None:
            self.color_history[pid].append(color_v)
            self._update_global_color_stats(color_v)
        if face_v is not None:
            self.face_embedding_history[pid].append(face_v)
        await self.publish_id(pid, gender, race, age, color_v, feat_v, first_seen, face_v)
        return pid

    async def update_person(self, person_id, gender, race, age, body_color, feature, face_embedding):
        await self._ensure_db_ready()
        feat_v = self.normalize(np.array(feature, dtype=np.float32)) if feature is not None else None
        color_v, _ = self.preprocess_color(np.array(body_color, dtype=np.float32)) if body_color is not None else (None, None)
        face_v = self.normalize(np.array(face_embedding, dtype=np.float32)) if face_embedding is not None else None

        gender = self._ensure_scalar(gender)
        race = self._ensure_scalar(race)
        age = self._ensure_scalar(age)

        # Kiểm tra định dạng dữ liệu đầu vào
        for name, vec in [("feature", feat_v), ("color", color_v), ("face_embedding", face_v)]:
            if vec is not None and vec.ndim != 1:
                logger.error(f"[{self.device_id}] {name} không phải mảng 1 chiều: {vec.shape}")
                return

        async with self.db_lock:
            retries = 3
            for attempt in range(retries):
                try:
                    async with self.temp_db.execute('SELECT person_id FROM Persons WHERE person_id = ?', (person_id,)) as cursor:
                        exists = await cursor.fetchone()
                    if not exists:
                        logger.warning(f"[{self.device_id}] {person_id} không tồn tại, tạo trong TempDB")
                        await self.create_person_in_temp(person_id, gender, race, age, body_color, feature, face_embedding)
                        return

                    ts = asyncio.get_event_loop().time()
                    await self.temp_db.execute(
                        "INSERT INTO Detections(person_id, timestamp, camera_id, frame_id, device_id) VALUES (?, ?, ?, ?, ?)",
                        (person_id, ts, self.device_id, f'frame_{int(ts)}', self.device_id)
                    )

                    if feat_v is not None:
                        self.feature_history[person_id].append(feat_v)
                    if color_v is not None:
                        self.color_history[person_id].append(color_v)
                        self._update_global_color_stats(color_v)
                    if face_v is not None:
                        self.face_embedding_history[person_id].append(face_v)

                    # Tính giá trị trung bình từ lịch sử với kiểm tra định dạng
                    mv_feat = None
                    if self.feature_history[person_id]:
                        if not all(feat.ndim == 1 for feat in self.feature_history[person_id]):
                            logger.error(f"[{self.device_id}] Lịch sử đặc trưng không phải mảng 1 chiều cho {person_id}")
                            return
                        mv_feat = np.mean(list(self.feature_history[person_id]), axis=0)

                    mv_color = None
                    if self.color_history[person_id]:
                        if not all(color.ndim == 1 for color in self.color_history[person_id]):
                            logger.error(f"[{self.device_id}] Lịch sử màu sắc không phải mảng 1 chiều cho {person_id}")
                            return
                        mv_color = np.mean(list(self.color_history[person_id]), axis=0)

                    mv_face = None
                    if self.face_embedding_history[person_id]:
                        if not all(face.ndim == 1 for face in self.face_embedding_history[person_id]):
                            logger.error(f"[{self.device_id}] Lịch sử face_embedding không phải mảng 1 chiều cho {person_id}")
                            return
                        mv_face = list(self.face_embedding_history[person_id])[-1]  # Lấy giá trị cuối cùng

                    if self.use_vec:
                        async with self.temp_db.execute("SELECT person_id FROM PersonVector WHERE person_id=?", (person_id,)) as cur:
                            exists = await cur.fetchone()
                        if exists:
                            await self.temp_db.execute(
                                "UPDATE PersonVector SET feature_mean=vec_f32(?), body_color_mean=vec_f32(?) WHERE person_id=?",
                                (mv_feat.tobytes() if mv_feat is not None else None, mv_color.tobytes() if mv_color is not None else None, person_id)
                            )
                        else:
                            await self.temp_db.execute(
                                "INSERT INTO PersonVector(person_id, feature_mean, body_color_mean) VALUES(?, vec_f32(?), vec_f32(?))",
                                (person_id, mv_feat.tobytes() if mv_feat is not None else None, mv_color.tobytes() if mv_color is not None else None)
                            )
                        if mv_face is not None:
                            async with self.temp_db.execute("SELECT person_id FROM FaceVector WHERE person_id=?", (person_id,)) as cur:
                                exists = await cur.fetchone()
                            if exists:
                                await self.temp_db.execute(
                                    "UPDATE FaceVector SET face_embedding=vec_f32(?) WHERE person_id=?",
                                    (mv_face.tobytes(), person_id)
                                )
                            else:
                                await self.temp_db.execute(
                                    "INSERT INTO FaceVector(person_id, face_embedding) VALUES(?, vec_f32(?))",
                                    (person_id, mv_face.tobytes())
                                )
                            await self.temp_db.execute(
                                "INSERT INTO FaceVectorHistory(person_id, face_embedding, timestamp) VALUES(?, vec_f32(?), ?)",
                                (person_id, mv_face.tobytes(), ts)
                            )
                    else:
                        await self.temp_db.execute(
                            "UPDATE Persons SET feature_mean=?, body_color_mean=? WHERE person_id=?",
                            (mv_feat.tobytes() if mv_feat is not None else None, mv_color.tobytes() if mv_color is not None else None, person_id)
                        )

                    meta_updates = []
                    params = []
                    if age is not None:
                        meta_updates.append("age = ?")
                        params.append(age)
                    if gender is not None:
                        meta_updates.append("gender = ?")
                        params.append(gender)
                    if race is not None:
                        meta_updates.append("race = ?")
                        params.append(race)
                    if meta_updates:
                        sql = f"UPDATE Face SET {', '.join(meta_updates)} WHERE person_id = ?"
                        params.append(person_id)
                        await self.temp_db.execute(sql, params)

                    await self.temp_db.commit()
                    break
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi cập nhật (lần {attempt + 1}/{retries}): {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise

        self.update_counts[person_id] += 1
        if self.update_counts[person_id] % 5 == 0 and mv_feat is not None and mv_color is not None and mv_face is not None:
            await self.publish_id(person_id, gender, race, age, mv_color, mv_feat, self.first_seen.get(person_id, asyncio.get_event_loop().time()), mv_face)

    ### Xử lý ID
    async def process_id(self, gender, race, age, body_color, features_person, face_embedding):
        gender = self._ensure_scalar(gender)
        race = self._ensure_scalar(race)
        age = self._ensure_scalar(age)
        matched_id = await self.match_id(gender, race, age, body_color, features_person, face_embedding)
        if matched_id:
            await self.update_person(matched_id, gender, race, age, body_color, features_person, face_embedding)
            if matched_id in self.recently_synced:
                self.recently_synced[matched_id] += 1
            logger.info(f"[{self.device_id}] Tìm thấy kết hợp cục bộ cho {matched_id}")
            return matched_id

        request_id = str(uuid.uuid4())
        sim_scores = {}
        self.pending_requests[request_id] = sim_scores
        self.person_buffer[request_id] = (gender, race, age, body_color, features_person, face_embedding)
        await self.publish_remote_request(request_id, gender, race, age, body_color, features_person, face_embedding)

        await asyncio.sleep(self.request_timeout)
        if request_id in self.pending_requests:
            sim_scores = self.pending_requests.pop(request_id)
            gender, race, age, body_color, feature, face_embedding = self.person_buffer.pop(request_id)
            if sim_scores:
                best_id = max(sim_scores, key=lambda x: sim_scores[x][0])
                best_sim, best_first_seen = sim_scores[best_id]
                if best_sim >= self.remote_feature_threshold:
                    async with self.db_lock:
                        async with self.temp_db.execute('SELECT person_id FROM Persons WHERE person_id = ?', (best_id,)) as cur:
                            exists = await cur.fetchone()
                    if exists:
                        await self.update_person(best_id, gender, race, age, body_color, feature, face_embedding)
                        self.recently_synced[best_id] += 1
                        logger.info(f"[{self.device_id}] Tìm thấy kết hợp từ xa cho {best_id} với độ tương đồng {best_sim}")
                    else:
                        await self.create_person_in_temp(best_id, gender, race, age, body_color, feature, face_embedding, first_seen=best_first_seen)
                        self.recently_synced[best_id] = 0
                        logger.info(f"[{self.device_id}] Đã tạo person với best_id từ xa {best_id} với độ tương đồng {best_sim}")
                    return best_id
            
            matched_id = await self.match_id(gender, race, age, body_color, feature, face_embedding)
            if matched_id:
                await self.update_person(matched_id, gender, race, age, body_color, feature, face_embedding)
                logger.info(f"[{self.device_id}] Tìm thấy kết hợp cục bộ sau timeout cho {matched_id}")
                return matched_id
            
            logger.info(f"[{self.device_id}] Không tìm thấy kết hợp phù hợp, tạo person mới")
            return await self.create_person(gender, race, age, body_color, feature, face_embedding)

    async def publish_remote_request(self, request_id, gender, race, age, body_color, feature, face_embedding):
        gender = self._ensure_scalar(gender)
        race = self._ensure_scalar(race)
        age = self._ensure_scalar(age)
        await self._ensure_rabbitmq_ready()
        if not self.channel or self.channel.is_closed:
            await self.setup_rabbitmq()
        try:
            message_body = json.dumps({
                "request_id": request_id,
                "gender": gender,
                "race": race,
                "age": age,
                "body_color": body_color.tolist() if body_color is not None else None,
                "feature": feature.tolist() if feature is not None else None,
                "face_embedding": face_embedding.tolist() if face_embedding is not None else None,
                "device_id": self.device_id,
                "timestamp": asyncio.get_event_loop().time()
            }).encode()
            message = aio_pika.Message(
                body=message_body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT, reply_to=self.response_queue.name
            )
            await self.remote_requests_exchange.publish(message, routing_key='')
            logger.info(f"[{self.device_id}] Đã gửi yêu cầu từ xa với request_id {request_id}")
        except Exception as e:
            logger.error(f"[{self.device_id}] Lỗi gửi yêu cầu từ xa: {str(e)}")
            await self.setup_rabbitmq()

    async def consume_remote_requests(self):
        await self._rabbitmq_ready.wait()
        while True:
            try:
                if not self.channel or self.channel.is_closed or not self.remote_requests_queue:
                    logger.warning(f"[{self.device_id}] Consumer yêu cầu từ xa chưa sẵn sàng, đang kết nối lại...")
                    await self.setup_rabbitmq()
                logger.info(f"[{self.device_id}] Bắt đầu tiêu thụ yêu cầu từ xa")
                async with self.remote_requests_queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            try:
                                request = json.loads(message.body.decode())
                                device_id = request.get("device_id")
                                if device_id == self.device_id:
                                    continue

                                request_id = request['request_id']
                                gender = self._ensure_scalar(request.get('gender'))
                                race = self._ensure_scalar(request.get('race'))
                                age = self._ensure_scalar(request.get('age'))
                                body_color = np.array(request['body_color'], dtype=np.float32) if request.get('body_color') else None
                                feature = np.array(request['feature'], dtype=np.float32) if request.get('feature') else None
                                face_embedding = np.array(request['face_embedding'], dtype=np.float32) if request.get('face_embedding') else None

                                logger.info(f"[{self.device_id}] Nhận yêu cầu từ xa request_id {request_id} từ {device_id}")
                                matched_id, sim_score = await self.match_id_synced(gender, race, age, body_color, feature, face_embedding)
                                response = {
                                    "request_id": request_id,
                                    "matched_id": matched_id,
                                    "similarity": sim_score if matched_id else 0.0,
                                    "first_seen": self.first_seen.get(matched_id, asyncio.get_event_loop().time()) if matched_id else None,
                                    "device_id": self.device_id
                                }
                                message_body = json.dumps(response).encode()
                                await self.channel.default_exchange.publish(
                                    aio_pika.Message(body=message_body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
                                    routing_key=message.reply_to
                                )
                                logger.info(f"[{self.device_id}] Đã phản hồi request_id {request_id} với matched_id={matched_id}")
                            except Exception as e:
                                logger.error(f"[{self.device_id}] Lỗi xử lý yêu cầu từ xa: {str(e)}")
            except Exception as e:
                logger.error(f"[{self.device_id}] Lỗi trong consumer yêu cầu từ xa: {str(e)}")
                await asyncio.sleep(5)

    ### Ghép ID
    async def match_id(self, new_gender, new_race, new_age, new_body_color, new_feature, new_face_embedding):
        await self._ensure_db_ready()
        start = time.time()

        if new_feature is None and new_body_color is None and new_face_embedding is None:
            logger.warning(f"[{self.device_id}] Không có dữ liệu để ghép.")
            return None

        # Chuẩn hóa dữ liệu đầu vào
        nfeat = self.normalize(np.array(new_feature, dtype=np.float32)) if new_feature is not None else None
        nc, _ = self.preprocess_color(np.array(new_body_color, dtype=np.float32)) if new_body_color is not None else (None, None)
        nface = self.normalize(np.array(new_face_embedding, dtype=np.float32)) if new_face_embedding is not None else None

        # Kiểm tra định dạng
        for name, vec in [("new_feature", nfeat), ("new_body_color", nc), ("new_face_embedding", nface)]:
            if vec is not None and vec.ndim != 1:
                logger.error(f"[{self.device_id}] {name} không phải mảng 1 chiều: {vec.shape}")
                return None

        new_gender = self._ensure_scalar(new_gender)
        new_race = self._ensure_scalar(new_race)
        new_age = self._ensure_scalar(new_age)

        async with self.db_lock:
            # Lấy danh sách ứng viên ban đầu
            async with self.temp_db.execute("SELECT person_id FROM Persons") as cur:
                candidates = [row[0] for row in await cur.fetchall()]
            logger.info(f"[{self.device_id}] Số ứng viên ban đầu: {len(candidates)}")

            if not candidates:
                return None

            # Bước 1: Lọc metadata linh hoạt
            placeholders = ",".join("?" for _ in candidates)
            query = f"""
                SELECT person_id FROM Face
                WHERE person_id IN ({placeholders})
                AND ((? IS NULL OR gender IS NULL) OR gender = ?)
                AND ((? IS NULL OR race IS NULL OR race = 'Unknown' OR ? = 'Unknown') OR race = ?)
                AND ((? IS NULL OR age IS NULL OR age = 'Unknown' OR ? = 'Unknown') OR age = ?)
            """
            params = candidates + [new_gender, new_gender, new_race, new_race, new_race, new_age, new_age, new_age]
            async with self.temp_db.execute(query, params) as cur:
                candidates = [row[0] for row in await cur.fetchall()]
            logger.info(f"[{self.device_id}] Lọc metadata → {len(candidates)} ứng viên")

            if not candidates:
                return None

            # Bước 2: Ưu tiên face_embedding
            best_face_pid = None
            best_face_sim = 0.0
            if nface is not None and self.use_vec:
                placeholders = ",".join("?" for _ in candidates)
                async with self.temp_db.execute(
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
                            logger.info(f"[{self.device_id}] Khớp face_embedding {best_face_pid} với sim={best_face_sim:.4f} trong {time.time()-start:.2f}s")
                            return best_face_pid

            # Bước 3: Lọc màu sắc cơ thể
            color_filtered_candidates = candidates
            if nc is not None and self.use_vec:
                placeholders = ",".join("?" for _ in candidates)
                k = min(self.top_k, len(candidates))
                sql = f"""
                    SELECT person_id, distance FROM PersonVector
                    WHERE person_id IN ({placeholders})
                    AND body_color_mean MATCH ? AND k = ?
                """
                async with self.temp_db.execute(sql, candidates + [nc.tobytes(), k]) as cur:
                    color_scores = [
                        (pid, 1 - (dist**2)/2)
                        for pid, dist in await cur.fetchall()
                        if 1 - (dist**2)/2 >= self.color_threshold
                    ]
                if color_scores:
                    color_filtered_candidates = [pid for pid, _ in color_scores]
                    logger.info(f"[{self.device_id}] Lọc màu sắc → {len(color_filtered_candidates)} ứng viên vượt ngưỡng")

            # Bước 4: So sánh đặc trưng
            if nfeat is not None:
                placeholders = ",".join("?" for _ in color_filtered_candidates)
                table = "PersonVector" if self.use_vec else "Persons"
                async with self.temp_db.execute(
                    f"SELECT person_id, feature_mean FROM {table} WHERE person_id IN ({placeholders})",
                    color_filtered_candidates
                ) as cur:
                    rows = await cur.fetchall()
                if rows:
                    ids, vecs = [], []
                    for pid, buf in rows:
                        if buf:
                            vec = np.frombuffer(buf, dtype=np.float32)
                            if vec.ndim == 1:
                                ids.append(pid)
                                vecs.append(vec)
                            else:
                                logger.error(f"[{self.device_id}] feature_mean cho {pid} không phải mảng 1 chiều")
                    if vecs:
                        vecs = np.stack(vecs, axis=0)
                        sims = cosine_similarity(vecs, nfeat.reshape(1, -1)).ravel()
                        best_idx = int(np.argmax(sims))
                        best_pid, best_sim = ids[best_idx], float(sims[best_idx])

                        if best_sim >= self.feature_threshold:
                            logger.info(f"[{self.device_id}] Khớp feature {best_pid} với sim={best_sim:.4f} trong {time.time()-start:.2f}s")
                            return best_pid

            # Bước 5: Độ tương đồng tổng hợp
            if nface is not None or nfeat is not None or nc is not None:
                # Bước 5: Độ tương đồng tổng hợp (vô hiệu hóa)
                logger.info(f"[{self.device_id}] Bước 5 (độ tương đồng tổng hợp) đã bị vô hiệu hóa")
                return None
                async with self.temp_db.execute("SELECT person_id FROM Persons") as cur:
                    all_candidates = [row[0] for row in await cur.fetchall()]
                best_combined_sim = 0.0
                best_combined_pid = None
                for pid in all_candidates:
                    face_sim = 0.0
                    if nface is not None and self.use_vec:
                        async with self.temp_db.execute("SELECT face_embedding FROM FaceVector WHERE person_id=?", (pid,)) as cur:
                            row = await cur.fetchone()
                        if row and row[0]:
                            face_vec = np.frombuffer(row[0], dtype=np.float32)
                            if face_vec.ndim == 1:
                                face_sim = cosine_similarity(face_vec.reshape(1, -1), nface.reshape(1, -1))[0][0]

                    color_sim = 0.0
                    if nc is not None:
                        async with self.temp_db.execute("SELECT body_color_mean FROM PersonVector WHERE person_id=?", (pid,)) as cur:
                            row = await cur.fetchone()
                        if row and row[0]:
                            color_vec = np.frombuffer(row[0], dtype=np.float32)
                            if color_vec.ndim == 1:
                                color_sim = cosine_similarity(color_vec.reshape(1, -1), nc.reshape(1, -1))[0][0]

                    feature_sim = 0.0
                    if nfeat is not None:
                        async with self.temp_db.execute("SELECT feature_mean FROM PersonVector WHERE person_id=?", (pid,)) as cur:
                            row = await cur.fetchone()
                        if row and row[0]:
                            feat_vec = np.frombuffer(row[0], dtype=np.float32)
                            if feat_vec.ndim == 1:
                                feature_sim = cosine_similarity(feat_vec.reshape(1, -1), nfeat.reshape(1, -1))[0][0]

                    if nface is None:
                        if feature_sim < 0.6 or color_sim < 0.5:
                            continue

                    combined_sim = (self.face_weight * face_sim if nface is not None else 0) + \
                                   (self.color_weight * color_sim if nc is not None else 0) + \
                                   (self.feature_weight * feature_sim if nfeat is not None else 0)

                    if combined_sim > best_combined_sim:
                        best_combined_sim = combined_sim
                        best_combined_pid = pid

                if best_combined_sim >= self.combined_threshold:
                    logger.info(f"[{self.device_id}] Khớp tổng hợp {best_combined_pid} với sim={best_combined_sim:.4f}")
                    return best_combined_pid

            return None

    async def match_id_synced(self, new_gender, new_race, new_age, new_body_color, new_feature, new_face_embedding):
        await self._ensure_db_ready()
        start = time.time()

        if new_feature is None and new_body_color is None and new_face_embedding is None:
            logger.warning(f"[{self.device_id}] Không có dữ liệu để ghép đồng bộ.")
            return None, 0.0

        nfeat = self.normalize(np.array(new_feature, dtype=np.float32)) if new_feature is not None else None
        nc, _ = self.preprocess_color(np.array(new_body_color, dtype=np.float32)) if new_body_color is not None else (None, None)
        nface = self.normalize(np.array(new_face_embedding, dtype=np.float32)) if new_face_embedding is not None else None

        # Kiểm tra định dạng
        for name, vec in [("new_feature", nfeat), ("new_body_color", nc), ("new_face_embedding", nface)]:
            if vec is not None and vec.ndim != 1:
                logger.error(f"[{self.device_id}] {name} không phải mảng 1 chiều: {vec.shape}")
                return None, 0.0

        new_gender = self._ensure_scalar(new_gender)
        new_race = self._ensure_scalar(new_race)
        new_age = self._ensure_scalar(new_age)

        async with self.db_lock:
            candidates = None
            if nface is not None and self.use_vec:
                sql = "SELECT person_id, distance FROM FaceVector WHERE face_embedding MATCH ? AND k = ?"
                async with self.temp_db.execute(sql, (nface.tobytes(), 3)) as cur:
                    face_scores = [(pid, 1 - (dist**2)/2) for pid, dist in await cur.fetchall() if 1 - (dist**2)/2 >= self.face_threshold]
                logger.info(f"[{self.device_id}] Lọc face_embedding đồng bộ → {len(face_scores)} ứng viên")
                if not face_scores:
                    return None, 0.0
                candidates = [pid for pid, _ in face_scores]
            else:
                async with self.temp_db.execute("SELECT person_id FROM Persons") as cur:
                    candidates = [row[0] for row in await cur.fetchall()]
                logger.info(f"[{self.device_id}] Không có face_embedding đồng bộ → {len(candidates)} ứng viên")

            if candidates:
                placeholders = ",".join("?" for _ in candidates)
                query = f"""
                    SELECT person_id FROM Face
                    WHERE person_id IN ({placeholders})
                    AND (age IS NULL OR ? IS NULL OR age = ?)
                    AND (gender IS NULL OR ? IS NULL OR gender = ?)
                    AND (race IS NULL OR ? IS NULL OR race = ?)
                """
                params = candidates + [new_age, new_age, new_gender, new_gender, new_race, new_race]
                async with self.temp_db.execute(query, params) as cur:
                    candidates = [row[0] for row in await cur.fetchall()]
                logger.info(f"[{self.device_id}] Lọc metadata đồng bộ → {len(candidates)} ứng viên")
                if not candidates:
                    return None, 0.0

            if nc is not None and self.use_vec and candidates:
                placeholders = ",".join("?" for _ in candidates)
                k = min(self.top_k, len(candidates))
                sql = f"""
                    SELECT person_id, distance FROM PersonVector
                    WHERE person_id IN ({placeholders})
                    AND body_color_mean MATCH ? AND k = ?
                """
                async with self.temp_db.execute(sql, candidates + [nc.tobytes(), k]) as cur:
                    color_scores = [(pid, 1 - (dist**2)/2) for pid, dist in await cur.fetchall() if 1 - (dist**2)/2 >= self.color_threshold]
                logger.info(f"[{self.device_id}] Lọc body_color_mean đồng bộ → {len(color_scores)} ứng viên")
                if not color_scores:
                    return None, 0.0
                candidates = [pid for pid, _ in color_scores]

            if nfeat is not None and candidates:
                placeholders = ",".join("?" for _ in candidates)
                table = "PersonVector" if self.use_vec else "Persons"
                async with self.temp_db.execute(
                    f"SELECT person_id, feature_mean FROM {table} WHERE person_id IN ({placeholders})",
                    candidates
                ) as cur:
                    rows = await cur.fetchall()
                if not rows:
                    return None, 0.0
                ids, vecs = [], []
                for pid, buf in rows:
                    if buf:
                        vec = np.frombuffer(buf, dtype=np.float32)
                        if vec.ndim == 1:
                            ids.append(pid)
                            vecs.append(vec)
                        else:
                            logger.error(f"[{self.device_id}] feature_mean cho {pid} không phải mảng 1 chiều")
                if vecs:
                    vecs = np.stack(vecs, axis=0)
                    sims = cosine_similarity(vecs, nfeat.reshape(1, -1)).ravel()
                    best_idx = int(np.argmax(sims))
                    best_pid, best_sim = ids[best_idx], float(sims[best_idx])

                    if best_sim < self.remote_feature_threshold:
                        logger.info(f"[{self.device_id}] pid={best_pid} sim={best_sim:.4f} dưới ngưỡng đồng bộ {self.remote_feature_threshold}")
                        return None, best_sim

                    logger.info(f"[{self.device_id}] Ghép đồng bộ {best_pid} sim={best_sim:.4f} trong {time.time()-start:.2f}s")
                    return best_pid, best_sim
            return None, 0.0

    ### Chuyển dữ liệu sang MainDB
    async def move_to_main_db(self, person_id):
        async with self.db_lock:
            retries = 3
            for attempt in range(retries):
                try:
                    async with self.temp_db.execute('SELECT * FROM Persons WHERE person_id = ?', (person_id,)) as cursor:
                        person_data = await cursor.fetchone()
                    if not person_data:
                        logger.warning(f"[{self.device_id}] Person_id {person_id} không tồn tại trong TempDB")
                        return

                    async with self.main_db.execute('SELECT person_id FROM Persons WHERE person_id = ?', (person_id,)) as cursor:
                        exists = await cursor.fetchone()
                    if exists:
                        logger.warning(f"[{self.device_id}] Person_id {person_id} đã tồn tại trong MainDB")
                        return

                    await self.main_db.execute('INSERT INTO Persons VALUES (?, ?)', person_data[:2])

                    async with self.temp_db.execute('SELECT * FROM Detections WHERE person_id = ?', (person_id,)) as cursor:
                        detections = await cursor.fetchall()
                    if detections:
                        await self.main_db.executemany('INSERT INTO Detections VALUES (?, ?, ?, ?, ?)', detections)

                    async with self.temp_db.execute('SELECT * FROM Face WHERE person_id = ?', (person_id,)) as cursor:
                        faces = await cursor.fetchall()
                    if faces:
                        await self.main_db.executemany('INSERT INTO Face VALUES (?, ?, ?, ?)', faces)

                    if self.use_vec:
                        async with self.temp_db.execute('SELECT * FROM PersonVector WHERE person_id = ?', (person_id,)) as cursor:
                            vector_data = await cursor.fetchone()
                        if vector_data:
                            await self.main_db.execute(
                                'INSERT INTO PersonVector (person_id, feature_mean, body_color_mean) VALUES (?, vec_f32(?), vec_f32(?))',
                                vector_data
                            )
                        async with self.temp_db.execute('SELECT * FROM FaceVector WHERE person_id = ?', (person_id,)) as cursor:
                            face_vector_data = await cursor.fetchone()
                        if face_vector_data:
                            await self.main_db.execute(
                                'INSERT INTO FaceVector (person_id, face_embedding) VALUES (?, vec_f32(?))',
                                face_vector_data
                            )
                    else:
                        await self.main_db.execute(
                            "UPDATE Persons SET feature_mean=?, body_color_mean=? WHERE person_id=?",
                            (person_data[2], person_data[3], person_id) if len(person_data) > 2 else (None, None, person_id)
                        )

                    await self.main_db.commit()
                    logger.info(f"[{self.device_id}] Đã chuyển {person_id} sang MainDB")
                    break
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi chuyển (lần {attempt + 1}/{retries}): {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise

    async def periodic_move_to_main_db(self):
        while True:
            await asyncio.sleep(300)
            async with self.db_lock:
                try:
                    async with self.temp_db.execute('''
                        SELECT person_id, COUNT(*) as detection_count
                        FROM Detections
                        GROUP BY person_id
                        HAVING detection_count >= 5
                    ''') as cursor:
                        persons_to_move = await cursor.fetchall()
                    for person_id, count in persons_to_move:
                        logger.info(f"[{self.device_id}] Đang chuyển {person_id} với {count} detections sang MainDB")
                        await self.move_to_main_db(person_id)
                except Exception as e:
                    logger.error(f"[{self.device_id}] Lỗi chuyển định kỳ: {str(e)}")

    ### Đóng kết nối
    async def _ensure_db_ready(self):
        while self.temp_db is None or self.main_db is None:
            await asyncio.sleep(0.1)

    async def _ensure_rabbitmq_ready(self):
        await self._rabbitmq_ready.wait()

    async def close(self):
        if self.temp_db:
            await self.temp_db.close()
            self.temp_db = None
        if self.main_db:
            await self.main_db.close()
            self.main_db = None
        if self.connection and not self.connection.is_closed:
            await self.connection.close() 
            self.connection = None
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    async def test():
        manager = ReIDManager(device_id="rpi", rabbitmq_url="amqp://new_user_rpi:123456@192.168.1.15/", request_timeout=3, wal_checkpoint_interval=300)
        await asyncio.sleep(5)
        await manager.close()
    asyncio.run(test())