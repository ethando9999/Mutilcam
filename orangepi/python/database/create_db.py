# file: python/database/create_db.py

import os
import aiosqlite
from contextlib import asynccontextmanager

# Import các hằng số từ file config
import config 
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Lấy các giá trị từ module config đã import
vec_extension_path: str = config.VEC0_PATH
FEATURE_DIMENSIONS: int = config.FEATURE_DIMENSIONS
FACE_DIMENSIONS: int = config.FACE_DIMENSIONS

async def create_database(
    db_path: str = "database.db",
) -> bool:
    """
    Khởi tạo SQLite database:
      - Tự động tạo thư mục cha nếu chưa tồn tại.
      - Nạp extension vec0.
      - Tạo schema cho các bảng.
    """
    try:
        # --- TỐI ƯU & SỬA LỖI ---
        # Lấy đường dẫn thư mục từ đường dẫn đầy đủ của file database
        db_directory = os.path.dirname(db_path)

        # Kiểm tra nếu thư mục không tồn tại, thì tạo nó một cách đệ quy
        if db_directory and not os.path.exists(db_directory):
            logger.info(f"Thư mục database '{db_directory}' không tồn tại. Đang tạo mới...")
            os.makedirs(db_directory, exist_ok=True)
        # --- KẾT THÚC SỬA LỖI ---

        # Kiểm tra sự tồn tại của file extension vec0
        if not os.path.exists(vec_extension_path):
            raise FileNotFoundError(f"vec0 extension not found at {vec_extension_path}")

        # Kết nối tới database. Nếu file chưa có, nó sẽ được tạo.
        async with aiosqlite.connect(db_path) as db:
            # Nạp extension vec0
            await db.enable_load_extension(True)
            await db.load_extension(vec_extension_path)
            
            version = (await (await db.execute('SELECT vec_version()')).fetchone())[0]
            logger.info(f"✅ Loaded sqlite-vec version {version}")

            # Thiết lập các PRAGMA để tối ưu hiệu suất và đảm bảo tính toàn vẹn
            await db.executescript( 
                """
                PRAGMA journal_mode  = WAL;
                PRAGMA synchronous   = NORMAL;
                PRAGMA foreign_keys  = ON;
                PRAGMA temp_store    = MEMORY;
                PRAGMA cache_size    = -20000;
                """
            )

            # --- Tạo schema chính ---
            # Sử dụng executescript cho phép chạy nhiều câu lệnh SQL một lúc
            await db.executescript(
                f"""
                -- Bảng gốc
                DROP TABLE IF EXISTS Persons;
                CREATE TABLE Persons (
                    person_id   TEXT PRIMARY KEY
                );

                -- Bảng ảo embedding
                DROP TABLE IF EXISTS PersonsVec;
                CREATE VIRTUAL TABLE PersonsVec USING vec0 (
                    person_id        TEXT PRIMARY KEY,
                    feature_mean     float[{FEATURE_DIMENSIONS}] DISTANCE cosine,
                    body_color_mean  float[51]  DISTANCE cosine
                );

                DROP TABLE IF EXISTS FaceVector;
                CREATE VIRTUAL TABLE IF NOT EXISTS FaceVector USING vec0 (
                    person_id       TEXT PRIMARY KEY,
                    face_embedding  float[{FACE_DIMENSIONS}] DISTANCE cosine
                );

                -- Trigger: tự động xóa vector khi person bị xóa
                DROP TRIGGER IF EXISTS trg_delete_person_vec;
                CREATE TRIGGER trg_delete_person_vec 
                AFTER DELETE ON Persons
                FOR EACH ROW
                BEGIN
                    DELETE FROM PersonsVec WHERE person_id = OLD.person_id;
                    DELETE FROM FaceVector WHERE person_id = OLD.person_id;
                END;

                -- Metadata
                DROP TABLE IF EXISTS PersonsMeta;
                CREATE TABLE PersonsMeta (
                    person_id    TEXT PRIMARY KEY,
                    age          TEXT,
                    gender       TEXT,
                    race         TEXT,
                    height_mean  REAL,
                    FOREIGN KEY (person_id) REFERENCES Persons(person_id) ON DELETE CASCADE
                );

                -- Cameras & Frames
                CREATE TABLE IF NOT EXISTS Cameras (
                    camera_id   TEXT PRIMARY KEY,
                    location    TEXT,
                    resolution  TEXT,
                    model       TEXT
                );
                CREATE TABLE IF NOT EXISTS Frames (
                    frame_id    TEXT PRIMARY KEY,
                    timestamp   REAL NOT NULL,
                    camera_id   TEXT NOT NULL,
                    frame_path  TEXT,
                    FOREIGN KEY (camera_id) REFERENCES Cameras(camera_id) ON DELETE SET NULL
                );

                -- Detections & Appearance
                CREATE TABLE IF NOT EXISTS Detections (
                    detection_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id           TEXT NOT NULL,
                    timestamp           REAL NOT NULL,
                    camera_id           TEXT,
                    frame_id            TEXT,
                    emotion             TEXT,
                    confidence_emotion  REAL,
                    bbox                TEXT,
                    FOREIGN KEY (person_id) REFERENCES Persons(person_id) ON DELETE CASCADE,
                    FOREIGN KEY (camera_id) REFERENCES Cameras(camera_id) ON DELETE SET NULL,
                    FOREIGN KEY (frame_id)  REFERENCES Frames(frame_id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS Appearance (
                    appearance_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id    INTEGER NOT NULL,
                    person_id       TEXT NOT NULL,
                    attire_top      TEXT,
                    attire_bottom   TEXT,
                    footwear        TEXT,
                    accessories     TEXT,
                    color_palette   TEXT,
                    confidence      REAL,
                    FOREIGN KEY (detection_id) REFERENCES Detections(detection_id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id)    REFERENCES Persons(person_id) ON DELETE CASCADE
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_meta_age_gender_race ON PersonsMeta(age, gender, race);
                CREATE INDEX IF NOT EXISTS idx_det_person     ON Detections(person_id);
                CREATE INDEX IF NOT EXISTS idx_det_camera     ON Detections(camera_id);
                CREATE INDEX IF NOT EXISTS idx_det_time       ON Detections(timestamp);
                CREATE INDEX IF NOT EXISTS idx_app_person     ON Appearance(person_id);
                CREATE INDEX IF NOT EXISTS idx_app_detection  ON Appearance(detection_id);
                CREATE INDEX IF NOT EXISTS idx_frames_camera  ON Frames(camera_id);
                CREATE INDEX IF NOT EXISTS idx_frames_time    ON Frames(timestamp);
                """
            )

            # --- Tạo schema cho các bảng FeatureBank và FaceIDBank ---
            await db.executescript(
                f"""
                -- FeatureBank metadata
                DROP TABLE IF EXISTS FeatureBank;
                CREATE TABLE FeatureBank (
                    feature_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id     TEXT    NOT NULL,
                    inserted_time REAL    NOT NULL DEFAULT (strftime('%s','now')),
                    FOREIGN KEY(person_id) REFERENCES Persons(person_id) ON DELETE CASCADE
                );

                -- Trigger giới hạn 20 vectors metadata
                DROP TRIGGER IF EXISTS trg_limit_featurebank;
                CREATE TRIGGER trg_limit_featurebank
                AFTER INSERT ON FeatureBank
                WHEN (SELECT COUNT(*) FROM FeatureBank WHERE person_id = NEW.person_id) > 20
                BEGIN
                    DELETE FROM FeatureBank
                    WHERE feature_id IN (
                        SELECT feature_id FROM FeatureBank WHERE person_id = NEW.person_id
                        ORDER BY inserted_time ASC LIMIT ((SELECT COUNT(*) FROM FeatureBank WHERE person_id = NEW.person_id) - 20)
                    );
                END;

                -- FeatureBank virtual vectors
                DROP TABLE IF EXISTS FeatureBankVec;
                CREATE VIRTUAL TABLE FeatureBankVec USING vec0 (
                    feature_id   INTEGER PRIMARY KEY,
                    person_id    TEXT    NOT NULL,
                    feature_vec  float[{FEATURE_DIMENSIONS}] DISTANCE cosine
                );

                -- Trigger xóa vector tương ứng
                DROP TRIGGER IF EXISTS trg_featbankvec_delete;
                CREATE TRIGGER trg_featbankvec_delete
                AFTER DELETE ON FeatureBank
                FOR EACH ROW BEGIN
                    DELETE FROM FeatureBankVec WHERE feature_id = OLD.feature_id;
                END;

                -- FaceIDBank metadata
                DROP TABLE IF EXISTS FaceIDBank;
                CREATE TABLE FaceIDBank (
                    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id     TEXT    NOT NULL,
                    inserted_time REAL    NOT NULL DEFAULT (strftime('%s','now')),
                    FOREIGN KEY(person_id) REFERENCES Persons(person_id) ON DELETE CASCADE
                );

                -- Trigger giới hạn 20 face vectors metadata
                DROP TRIGGER IF EXISTS trg_limit_faceidbank;
                CREATE TRIGGER trg_limit_faceidbank
                AFTER INSERT ON FaceIDBank
                WHEN (SELECT COUNT(*) FROM FaceIDBank WHERE person_id = NEW.person_id) > 20
                BEGIN
                    DELETE FROM FaceIDBank
                    WHERE face_id IN (
                        SELECT face_id FROM FaceIDBank WHERE person_id = NEW.person_id
                        ORDER BY inserted_time ASC LIMIT ((SELECT COUNT(*) FROM FaceIDBank WHERE person_id = NEW.person_id) - 20)
                    );
                END;

                -- FaceIDBank virtual vectors
                DROP TABLE IF EXISTS FaceIDBankVec;
                CREATE VIRTUAL TABLE FaceIDBankVec USING vec0 (
                    face_id      INTEGER PRIMARY KEY,
                    person_id    TEXT    NOT NULL,
                    face_vec     float[{FACE_DIMENSIONS}] DISTANCE cosine
                );

                -- Trigger xóa vector tương ứng
                DROP TRIGGER IF EXISTS trg_faceidbankvec_delete;
                CREATE TRIGGER trg_faceidbankvec_delete
                AFTER DELETE ON FaceIDBank
                FOR EACH ROW BEGIN
                    DELETE FROM FaceIDBankVec WHERE face_id = OLD.face_id;
                END;
                """
            )

            await db.commit()
            logger.info(f"Database initialized at {db_path} with vec0 extension.")
            return True

    except Exception as e:
        logger.exception(f"create_database() failed: {e}")
        raise

# Các hàm helper còn lại
async def initialize_db(db_path='temp.db', vec_extension_path='/usr/local/lib/vec0.so'):
    """Khởi tạo cơ sở dữ liệu nếu chưa tồn tại."""
    try:
        # Hàm create_database đã nhận đủ thông tin từ config
        await create_database(db_path)
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise

@asynccontextmanager
async def open_db(db_path="temp.db", vec_extension_path="/usr/local/lib/vec0.so"):
    """Context manager để mở kết nối database."""
    # Đoạn code này nên được sửa để nhất quán với create_database
    db_directory = os.path.dirname(db_path)
    if db_directory and not os.path.exists(db_directory):
        os.makedirs(db_directory, exist_ok=True)
        
    if not os.path.exists(vec_extension_path):
        raise FileNotFoundError(f"{vec_extension_path} not found")

    db = await aiosqlite.connect(db_path)
    await db.enable_load_extension(True)
    try:
        await db.load_extension(vec_extension_path)
    except aiosqlite.Error as e:
        if "already loaded" not in str(e).lower():
            await db.close()
            raise
    try:
        yield db
    finally:
        await db.close()
