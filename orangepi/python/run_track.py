# file: python/run_track.py (Phiên bản cuối cùng)

import argparse
import asyncio
import os
import psutil

# Import các module của dự án
from database.create_db import create_database
from track_local.byte_track import TrackingManager
import config
# << SỬA Ở ĐÂY: Import class SocketSender >>
from core.socket_sender import SocketSender

# Thiết lập logging
from utils.logging_python_orangepi import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)


async def monitor_queue(queue: asyncio.Queue, queue_name: str):
    # ... (hàm này không đổi)
    while True:
        try:
            max_size_str = f"/{queue.maxsize}" if queue.maxsize > 0 else ""
            logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}{max_size_str}")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break

async def monitor_system():
    # ... (hàm này không đổi)
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            logger.info(f"Tải hệ thống - CPU: {cpu_percent:.1f}%, Bộ nhớ: {memory.percent:.1f}%")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break

async def main(args):
    logger.info("Khởi động ứng dụng chính…")

    device_id = args.device_id
    logger.info(f"Đang chạy ở chế độ device_id = {device_id}")

    if "opi" in device_id:
        from core.put_RGBD import start_putter 
        from core.processing_track1 import start_processor
        ID_CONFIG = config.OPI_CONFIG
    else:
        logger.critical(f"device_id không hợp lệ: '{device_id}'. Chương trình dừng lại.") 
        return

    # --- Khởi tạo cơ sở dữ liệu ---
    db_path = ID_CONFIG.get("db_path") 
    if args.new_db:
        logger.info(f"Cờ --new-db được bật. Đang tạo lại cơ sở dữ liệu tại '{db_path}'.")
        for suffix in ["", "-shm", "-wal"]:
            if os.path.exists(f"{db_path}{suffix}"): os.remove(f"{db_path}{suffix}")
        await create_database(db_path=db_path)
    elif not os.path.exists(db_path):
        logger.info(f"Cơ sở dữ liệu không tồn tại tại '{db_path}'. Đang tạo mới...")
        await create_database(db_path=db_path)

    # --- Khởi tạo các hàng đợi (Queues) ---
    frame_queue = asyncio.Queue(maxsize=5)
    processing_queue = asyncio.Queue(maxsize=200) 
    people_count_queue = asyncio.Queue(maxsize=10)
    height_queue = asyncio.Queue(maxsize=10)
    track_profile_queue = asyncio.Queue(maxsize=100) # Chỉ cần hàng đợi profile

    all_tasks = []
    try:
        # 1. Khởi tạo Tracker
        tracker = TrackingManager(
            detection_queue=processing_queue,
            track_profile_queue=track_profile_queue
        )

        # 2. Khởi tạo các tác vụ Worker
        all_tasks.append(asyncio.create_task(start_putter(frame_queue, camera_id=ID_CONFIG.get("rgb_camera_id", 0)), name="PutterTask"))

        calib_path = ID_CONFIG.get("calib_file_path")
        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError(f"LỖI: File hiệu chỉnh không được tìm thấy tại '{calib_path}'.")
        all_tasks.append(asyncio.create_task(start_processor(
            frame_queue, processing_queue, people_count_queue, height_queue, calib_path
        ), name="ProcessorTask"))

        all_tasks.append(asyncio.create_task(tracker.run(), name="TrackerTask"))

        # 3. Khởi tạo các tác vụ Giao tiếp (WebSocket Senders)
        # all_tasks.append(asyncio.create_task(SocketSender(ID_CONFIG["SOCKET_COUNT_URI"], people_count_queue, "PeopleCountSender").run(), name="CountSender"))
        # all_tasks.append(asyncio.create_task(SocketSender(ID_CONFIG["SOCKET_HEIGHT_URI"], height_queue, "HeightSender").run(), name="HeightSender"))
        all_tasks.append(asyncio.create_task(SocketSender(ID_CONFIG["SOCKET_TRACK_COLOR_URI"], track_profile_queue, "TrackProfileSender").run(), name="TrackProfileSender"))
        
        # 4. Khởi tạo các tác vụ Giám sát
        all_tasks.append(asyncio.create_task(monitor_system(), name="SystemMonitor"))
        all_tasks.append(asyncio.create_task(monitor_queue(frame_queue, "Frame"), name="FrameQueueMonitor"))
        all_tasks.append(asyncio.create_task(monitor_queue(processing_queue, "Processing"), name="ProcQueueMonitor"))
        all_tasks.append(asyncio.create_task(monitor_queue(track_profile_queue, "TrackProfile"), name="TrackProfileQueueMonitor"))

        logger.info(f"Đã khởi tạo {len(all_tasks)} tác vụ. Bắt đầu vòng lặp chính.")
        await asyncio.gather(*all_tasks)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("Phát hiện tín hiệu dừng (Ctrl+C). Đang tắt chương trình...")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn ở cấp cao nhất: {e}", exc_info=True)
    finally:
        logger.info("Bắt đầu quá trình dọn dẹp tài nguyên...")
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        logger.info("Tất cả tài nguyên đã được dọn dẹp. Chương trình kết thúc.")

if __name__ == "__main__":
    # ... (phần này không đổi)
    parser = argparse.ArgumentParser(description="Chạy hệ thống ReID và Tracking đa luồng.")
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại cơ sở dữ liệu nếu tồn tại.")
    parser.add_argument("--device-id", type=str, default="opi", help="ID của thiết bị (phải chứa 'opi' để chạy).")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Chương trình đã bị ngắt bởi người dùng.") 