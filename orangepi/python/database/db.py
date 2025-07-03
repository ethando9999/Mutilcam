# utils/db.py
from contextlib import asynccontextmanager
import aiosqlite, os

@asynccontextmanager
async def open_db(db_path="temp.db",
                  vec_extension_path="/usr/local/lib/vec0.so"):
    if not os.path.exists(vec_extension_path):
        raise FileNotFoundError(f"{vec_extension_path} not found")

    db = await aiosqlite.connect(db_path)
    await db.enable_load_extension(True)
    try:                       # nạp vec0
        await db.load_extension(vec_extension_path)
    except aiosqlite.Error as e:
        if "already loaded" not in str(e).lower():
            await db.close()
            raise
    try:
        yield db               # <-- dùng trong thân `async with`
    finally:
        await db.close()
