from database.log_similar import log_similar
from database.log_id_new import log_all_ids, log_detections
import asyncio
from config import DEVICE_ID_CONFIG_1 as ID_CONFIG
db_path = ID_CONFIG.get("db_path", "database.db") 

async def main():
    await log_all_ids(db_path)
    await log_detections(db_path)
    await log_similar(db_path)

if __name__ == "__main__":
    asyncio.run(main())