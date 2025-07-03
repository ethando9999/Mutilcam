import aiosqlite
import asyncio
import os
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

async def create_database(db_path='temp.db', vec_extension_path='/usr/local/lib/vec0.so', require_vec=False):
    """Tạo cơ sở dữ liệu SQLite với sqlite-vec hoặc fallback sang BLOB cho hệ thống ReID."""
    async with aiosqlite.connect(db_path) as db:
        # Kiểm tra sự tồn tại của tệp extension
        if not os.path.exists(vec_extension_path):
            logger.error(f"Extension file not found: {vec_extension_path}")
            if require_vec:
                raise Exception("sqlite-vec extension required but not found.")
            use_vec = False
        else:
            # Thử tải extension sqlite-vec
            try:
                await db.enable_load_extension(True)
                await db.load_extension(vec_extension_path)
                logger.info(f"Extension sqlite-vec loaded successfully from {vec_extension_path}.")
                # Kiểm tra xem vec0 có thực sự hoạt động không
                await db.execute("SELECT vec_version();")
                use_vec = True
            except Exception as e:
                logger.error(f"Failed to load or use sqlite-vec from {vec_extension_path}: {e}")
                if require_vec:
                    raise Exception("sqlite-vec extension required but failed to load or function.")
                logger.info("Falling back to BLOB due to vec0 module issue.")
                use_vec = False

        # Tối ưu hóa hiệu năng SQLite
        await db.execute('PRAGMA journal_mode=WAL;')
        await db.execute('PRAGMA synchronous=NORMAL;')
        await db.execute('PRAGMA foreign_keys=ON;')
        await db.execute('PRAGMA temp_store=MEMORY;')
        await db.execute('PRAGMA cache_size=-20000;')
        logger.info("SQLite PRAGMA settings applied.")

        # Bảng Persons
        await db.execute('''
            CREATE TABLE IF NOT EXISTS Persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame BLOB,
                frame_count INTEGER DEFAULT 0
            )
        ''')
        logger.info("Table Persons created or already exists.")

        # Bảng Detections
        await db.execute('''
            CREATE TABLE IF NOT EXISTS Detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                camera_id TEXT NOT NULL,
                frame_id TEXT,
                num_features INTEGER DEFAULT 0,
                num_colors INTEGER DEFAULT 0,
                FOREIGN KEY (person_id) REFERENCES Persons(person_id) ON DELETE CASCADE
            )
        ''')
        logger.info("Table Detections created or already exists.")

        if use_vec:
            try:
                await db.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS DetectionVectors USING vec0 (
                        detection_id     INTEGER PRIMARY KEY,
                        feature_mean     float[512],
                        body_color_mean  float[51],
                        body_color_mask  float[51]
                    )
                ''')
                logger.info("Created DetectionVectors virtual table with mask.")
            except Exception as e:
                logger.error(f"vec0 error: {e}, fallback to BLOB")
                use_vec = False

        if not use_vec:
            # danh sách các cột cần thêm vào Detections
            cols = ["feature_mean", "body_color_mean", "body_color_mask"]
            for col in cols:
                try:
                    await db.execute(f'ALTER TABLE Detections ADD COLUMN {col} BLOB;')
                    logger.info(f"Added column {col} to Detections.")
                except aiosqlite.OperationalError as e:
                    if "duplicate column name" in str(e):
                        logger.debug(f"Column {col} already exists, skip.")
                    else:
                        raise

        # Bảng Face
        await db.execute('''
            CREATE TABLE IF NOT EXISTS Face (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                age INTEGER,
                gender TEXT,
                emotion TEXT,
                race TEXT,
                FOREIGN KEY (detection_id) REFERENCES Detections(detection_id) ON DELETE CASCADE
            )
        ''')
        logger.info("Table Face created or already exists.")

        # Bảng Appearance
        await db.execute('''
            CREATE TABLE IF NOT EXISTS Appearance (
                appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                height_cm REAL,
                attire_top TEXT,
                attire_bottom TEXT,
                footwear TEXT,
                accessories TEXT,
                color_palette TEXT,
                FOREIGN KEY (detection_id) REFERENCES Detections(detection_id) ON DELETE CASCADE
            )
        ''')
        logger.info("Table Appearance created or already exists.")

        # Bảng Frames
        await db.execute('''
            CREATE TABLE IF NOT EXISTS Frames (
                frame_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                frame_path TEXT
            )
        ''')
        logger.info("Table Frames created or already exists.")

        # Bảng PublishStatus
        await db.execute('''
            CREATE TABLE IF NOT EXISTS PublishStatus (
                publish_id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                publish_time TEXT,
                status TEXT CHECK(status IN ('pending', 'sent', 'failed')),
                FOREIGN KEY (detection_id) REFERENCES Detections(detection_id) ON DELETE CASCADE
            )
        ''')
        logger.info("Table PublishStatus created or already exists.")

        # Tạo chỉ mục
        await db.execute('CREATE INDEX IF NOT EXISTS idx_detections_person_id ON Detections(person_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_detections_camera_id ON Detections(camera_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON Detections(timestamp)')
        logger.info("Indexes for Detections created or already exist.")

        await db.execute('CREATE INDEX IF NOT EXISTS idx_face_detection_id ON Face(detection_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_face_age ON Face(age)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_face_gender ON Face(gender)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_face_race ON Face(race)')
        logger.info("Indexes for Face created or already exist.")

        await db.execute('CREATE INDEX IF NOT EXISTS idx_appearance_detection_id ON Appearance(detection_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_appearance_height_cm ON Appearance(height_cm)')
        logger.info("Indexes for Appearance created or already exist.")

        await db.execute('CREATE INDEX IF NOT EXISTS idx_frames_camera_id ON Frames(camera_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON Frames(timestamp)')
        logger.info("Indexes for Frames created or already exist.")

        await db.execute('CREATE INDEX IF NOT EXISTS idx_publish_detection_id ON PublishStatus(detection_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_publish_status ON PublishStatus(status)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_publish_time ON PublishStatus(publish_time)')
        logger.info("Indexes for PublishStatus created or already exist.")

        # Kiểm tra bảng Detections
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Detections';")
        if not await cursor.fetchone():
            logger.error("Table Detections was not created successfully.")
            raise Exception("Failed to create Detections table")
        else:
            logger.info("Table Detections confirmed to exist.") 

        await db.commit()
        logger.info(f"Database created successfully at {db_path}")
        return use_vec  # Trả về trạng thái sử dụng sqlite-vec

async def initialize_db(db_path='temp.db', vec_extension_path='/usr/local/lib/vec0.so', require_vec=False):
    """Khởi tạo cơ sở dữ liệu nếu chưa tồn tại."""
    try:
        use_vec = await create_database(db_path, vec_extension_path, require_vec)
        return use_vec
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="Initialize SQLite database for ReID system.")
    parser.add_argument('--db_path', type=str, default='temp.db', help='Path to the database file.')
    args = parser.parse_args()
    asyncio.run(initialize_db(db_path=args.db_path))