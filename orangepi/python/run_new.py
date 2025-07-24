# file: python/run.py

import argparse
import asyncio
import os
import psutil

# Import các module của dự án
from database.create_db import create_database
from track_local.byte_track import TrackingManager
import config
from core.socket import start_socket_sender
from core.put_RGBD import start_putter
from core.processing_track_son import start_processor
from SMC.web3_table_client import TableAIClient

# Thiết lập logging
from utils.logging_python_orangepi import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)


async def monitor_queue(queue: asyncio.Queue, queue_name: str):
    """Giám sát kích thước hàng đợi."""
    while True:
        try:
            max_size_str = f"/{queue.maxsize}" if queue.maxsize > 0 else ""
            logger.info(f"Hàng đợi {queue_name}: {queue.qsize()}{max_size_str}")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break


async def monitor_system():
    """Giám sát tải CPU và bộ nhớ."""
    while True:
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            logger.info(f"Tải hệ thống - CPU: {cpu:.1f}%, MEM: {mem:.1f}%")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break


async def main(args):
    logger.info("=== Khởi động ứng dụng ReID & Tracking ===")
    device_id = args.device_id
    logger.info(f"Device ID: {device_id}")

    if "opi" not in device_id:
        logger.critical(f"device_id không hợp lệ: '{device_id}'. Dừng chương trình.")
        return
    ID_CFG = config.OPI_CONFIG

    # --- DB setup ---
    db_path = ID_CFG["db_path"]
    if args.new_db:
        logger.info(f"--new-db: xóa và tạo mới '{db_path}'")
        for suf in ("", "-shm", "-wal"):
            p = f"{db_path}{suf}"
            if os.path.exists(p): os.remove(p)
        await create_database(db_path=db_path)
    elif not os.path.exists(db_path):
        logger.info(f"Tạo DB mới tại '{db_path}'")
        await create_database(db_path=db_path)

    # --- Queues ---
    frame_q = asyncio.Queue(maxsize=5)
    proc_q  = asyncio.Queue(maxsize=200)
    count_q = asyncio.Queue(maxsize=1)
    height_q= asyncio.Queue(maxsize=1)
    track_q = asyncio.Queue(maxsize=2) 

    # --- TableAIClient (blockchain sender) ---
    try:
        table_client = TableAIClient(disable_tls=ID_CFG.get("DISABLE_TLS", True))
    except Exception as e:
        logger.error(f"Không thể khởi tạo TableAIClient: {e}")
        return

    tasks = []
    try:
        # 1. Putter (RGBD capture)
        cam_id = ID_CFG.get("rgb_camera_id", 0)
        tasks.append(
            asyncio.create_task(
                start_putter(frame_q, camera_id=cam_id),
                name="PutterTask"
            )
        )

        # 2. Processor (detection + height + count)
        calib = ID_CFG["calib_file_path"]
        if not os.path.exists(calib):
            raise FileNotFoundError(f"File calib không tìm thấy: '{calib}'")
        tasks.append(
            asyncio.create_task(
                start_processor(frame_q, proc_q, count_q, height_q, calib),  
                name="ProcessorTask"
            )
        )

        # 3. Tracker
        tracker = TrackingManager(detection_queue=proc_q, track_profile_queue=track_q)
        tasks.append(
            asyncio.create_task(tracker.run(), name="TrackerTask")
        )

        # 4. Socket senders
        tasks.append(asyncio.create_task(start_socket_sender(track_q, config.OPI_CONFIG["SOCKET_TRACK_COLOR_URI"])))

        # 5. TableAIClient sender: gửi chiều cao tối thiểu lên blockchain
        tasks.append(
            asyncio.create_task(
                table_client.run(height_q),
                name="TableAIClient"
            )
        )

        # 6. Giám sát
        tasks.extend([
            asyncio.create_task(monitor_system(), name="SystemMonitor"),
            asyncio.create_task(monitor_queue(frame_q, "Frame"), name="FrameMonitor"),
            asyncio.create_task(monitor_queue(proc_q,  "Processing"), name="ProcMonitor"),
            asyncio.create_task(monitor_queue(track_q, "TrackProfile"), name="TrackProfileMonitor"),
            asyncio.create_task(monitor_queue(height_q, "Height"), name="HeightMonitor"),
        ])

        logger.info(f"Khởi tạo {len(tasks)} tác vụ, bắt đầu chạy…")
        await asyncio.gather(*tasks)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("Đã nhận lệnh dừng, hủy tất cả tasks…")
    except Exception as e:
        logger.error(f"Lỗi cấp cao nhất: {e}", exc_info=True)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Chương trình kết thúc, dọn dẹp xong.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chạy hệ thống ReID & Tracking")
    p.add_argument("--new-db",   action="store_true", help="Tạo lại DB")
    p.add_argument("--device-id", type=str, default="opi", help="ID thiết bị (phải chứa 'opi')")
    args = p.parse_args()
    asyncio.run(main(args))
