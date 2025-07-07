import aiosqlite
import asyncio
import logging

# Thiết lập logger cơ bản để đảm bảo log hiển thị trên console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def check_table_exists(db, table_name):
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    async with db.execute(query, (table_name,)) as cursor:
        return await cursor.fetchone() is not None


async def create_database(db_path='database.db'):
    """Tạo database với hai bảng: PersonsVec (vector) và PersonsMeta (metadata)."""
    logger.info("Đang tạo lại database...")

    try:
        async with aiosqlite.connect(db_path) as db:
            # Bật hỗ trợ foreign key
            await db.execute("PRAGMA foreign_keys = ON;")

            # Xóa bảng cũ nếu có
            await db.execute("DROP TABLE IF EXISTS PersonsMeta;")

            # Tạo bảng metadata
            await db.execute("""
                CREATE TABLE PersonsMeta (
                    person_id    INTEGER PRIMARY KEY,
                    age          INTEGER,
                    gender       TEXT,
                    race         TEXT,
                    height_mean  REAL,
                    FOREIGN KEY(person_id) REFERENCES PersonsVec(person_id) ON DELETE CASCADE
                );
            """)

            await db.commit()
            logger.info("Tạo database thành công.")

    except Exception as e:
        logger.error(f"Lỗi khi tạo database: {e}")


async def log_all_ids(main_db_path='database.db'):
    """Log tất cả person_id từ PersonsVec và toàn bộ nội dung PersonsMeta."""
    logger.info("Bắt đầu log_all_ids...")

    try:
        async with aiosqlite.connect(main_db_path) as db:
            meta_exists = await check_table_exists(db, 'PersonsMeta')

            if not meta_exists:
                logger.warning("Thiếu bảng. Hãy đảm bảo đã gọi initialize_db() trước đó.")
                return

            # Log toàn bộ nội dung bảng PersonsMeta
            async with db.execute('SELECT * FROM PersonsMeta') as cursor:
                rows = [row async for row in cursor]
                logger.info("Dữ liệu PersonsMeta:")
                for row in rows:
                    logger.info(row)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý database: {e}")

async def log_detections(db_path='database.db'):
    """Log số lần xuất hiện của mỗi person_id trong bảng Detections."""
    logger.info("Bắt đầu log_detections...")

    try:
        async with aiosqlite.connect(db_path) as db:
            detections_exist = await check_table_exists(db, 'Detections')
            if not detections_exist:
                logger.warning("Bảng Detections không tồn tại.")
                return

            query = """
                SELECT person_id, COUNT(*) as count 
                FROM Detections 
                GROUP BY person_id 
                ORDER BY count DESC
            """

            async with db.execute(query) as cursor:
                results = [row async for row in cursor]
                if results:
                    logger.info("Số lần xuất hiện của các person_id:")
                    for person_id, count in results:
                        logger.info(f"person_id={person_id}, count={count}")
                else:
                    logger.info("Không có dữ liệu trong bảng Detections.") 

    except Exception as e:
        logger.error(f"Lỗi khi log Detections: {e}") 



if __name__ == "__main__":
    from orangepi.python.config_x import DEVICE_ID_CONFIG_1 as ID_CONFIG
    db_path = ID_CONFIG.get("db_path", "database.db") 
    asyncio.run(log_all_ids(db_path)) 