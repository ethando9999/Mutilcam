import ast
import logging
from typing import Any, List, Tuple
import asyncio

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from database.db import open_db

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

def _to_ndarray(raw: Any, dtype=np.float32) -> np.ndarray:
    """
    Convert whatever comes out of SQLite (list-literal str, JSON str, bytes, list…)
    into a 1-D NumPy array.
    """
    if raw is None:
        raise ValueError("Vector field is NULL")

    # If it’s already a (list | tuple | np.ndarray)
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=dtype)

    # If stored as bytes (e.g. vec0 BLOB)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return np.frombuffer(raw, dtype=dtype)

    # Otherwise assume it’s a string like “[0.1, 0.2, …]”
    if isinstance(raw, str):
        return np.asarray(ast.literal_eval(raw), dtype=dtype)

    raise TypeError(f"Unsupported vector format: {type(raw)}")


async def log_similar(db_path="temp.db",
                      vec_extension_path="/usr/local/lib/vec0.so",
                      top_k=5,
                      k_neighbors=50):
    logger.info("Bắt đầu log_similar…")
    try:
        async with open_db(db_path, vec_extension_path) as db:
            async with db.execute("""
                SELECT person_id, feature_mean, body_color_mean
                FROM PersonsVec
            """) as cur:
                records = [
                    (pid, _to_ndarray(f_vec), _to_ndarray(c_vec))
                    async for pid, f_vec, c_vec in cur
                ]

            if len(records) < 2:
                logger.info("Chỉ có 0-1 bản ghi – không thể tính độ tương đồng.")
                return

            ids         = np.array([r[0] for r in records])
            feature_mat = np.stack([r[1] for r in records])
            feat_sim    = cosine_similarity(feature_mat)

            for i, (pid, _, body_vec) in enumerate(records):
                # Truy vấn từ vec0 các láng giềng màu gần nhất
                sql = """
                    SELECT person_id, distance
                    FROM PersonsVec
                    WHERE person_id != ?
                    AND body_color_mean MATCH ? AND k = ?
                """
                async with db.execute(sql, [pid, body_vec.tobytes(), k_neighbors]) as cur:
                    rows = await cur.fetchall()

                color_scores = {
                    other_id: 1 - (dist ** 2) / 2
                    for other_id, dist in rows
                }

                combined_list = []
                for j, other_id in enumerate(ids):
                    if other_id == pid:
                        continue
                    feat = feat_sim[i, j]
                    col  = color_scores.get(other_id, 0.0)
                    avg  = (feat + col) / 2.0
                    combined_list.append((other_id, feat, col, avg))

                combined_list.sort(key=lambda t: t[3], reverse=True)
                top = combined_list[:top_k]

                logger.info(
                    f"person_id={pid} ─ Top-{top_k} similar IDs "
                    f"(avg_cosine = feature/body_color):"
                )
                for other_id, f_sim, c_sim, avg in top:
                    logger.info(
                        f"    ↳ {other_id} "
                        f"(feature={f_sim:.4f}, "
                        f"color={c_sim:.4f}, "
                        f"avg={avg:.4f})"
                    )

    except Exception as e:
        logger.exception(f"log_similar() failed: {e}")

if __name__ == "__main__":
    from config import ID_CONFIG
    db_path = ID_CONFIG.get("db_path", "database.db") 
    asyncio.run(log_similar(db_path)) 