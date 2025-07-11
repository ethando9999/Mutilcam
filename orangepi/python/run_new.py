import argparse
import asyncio
import os
import psutil

from database.create_db import create_database
from id.main_id_new import start_id, PersonReID
from utils.logging_python_orangepi import setup_logging, get_logger

# Thiết lập logging
setup_logging()
logger = get_logger(__name__)

async def monitor_queue(queue, queue_name):
    """Log queue size periodically."""
    while True:
        logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}/{getattr(queue, 'maxsize', 'N/A')}")
        await asyncio.sleep(5)

async def monitor_system():
    """Log CPU & RAM usage periodically."""
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        logger.info(f"Tải hệ thống - CPU: {cpu_percent:.1f}%, Bộ nhớ: {memory.percent:.1f}%")
        await asyncio.sleep(5)

async def main(args):
    """Entry point for asyncio event loop."""
    logger.info("Khởi động ứng dụng chính…")

    # Chọn cấu hình và module phù hợp
    device_id = args.device_id
    if "opi" in device_id:
        from core.put_RGBD import start_putter
        from core.processing_RGBD import start_processor
        from core.websocket_sender import start_sender_worker
        from config import OPI_CONFIG as ID_CONFIG
    elif "rpi" in device_id:
        from core.put_frame import start_putter
        from core.processing import start_processor
        from config import RPI_CONFIG as ID_CONFIG
    else:
        raise ValueError("device_id phải là 'rpi' hoặc 'opi'")

    logger.info(f"Đang chạy ở chế độ device_id = {device_id}")

    db_path = ID_CONFIG.get("db_path", "database.db")
    if args.new_db and os.path.exists(db_path): 
        os.remove(db_path)
        logger.info(f"Đã xóa DB cũ: {db_path}")

    if not os.path.exists(db_path):
        await create_database(db_path=db_path)
        logger.info(f"Đã tạo cơ sở dữ liệu mới: {db_path}")
    else:
        logger.info(f"Sử dụng cơ sở dữ liệu hiện có: {db_path}")

    # Khởi tạo các hàng đợi asyncio
    frame_queue = asyncio.Queue(maxsize=1000)
    processing_queue = asyncio.Queue(maxsize=200) # Queue cho Re-ID
    socket_queue = asyncio.Queue(maxsize=100)      # Queue cho WebSocket

    # Khởi tạo PersonReID
    valid_keys = ["output_dir", "feature_threshold", "color_threshold", "avg_threshold",
                  "top_k", "thigh_weight", "torso_weight", "feature_weight", "color_weight",
                  "temp_timeout", "min_detections", "merge_threshold", "face_threshold"]
    filtered_config = {k: ID_CONFIG[k] for k in valid_keys if k in ID_CONFIG}
    filtered_config.update({"device_id": device_id, "db_path": db_path})
    person_reid = PersonReID(**filtered_config)

    all_tasks = []
    try:
        # Task ghi nhận frame
        all_tasks.append(asyncio.create_task(start_putter(frame_queue, ID_CONFIG.get("rgb_camera_id", 0))))
        
        # Task xử lý frame (nhận frame, xuất ra processing_queue và socket_queue)
        all_tasks.append(asyncio.create_task(start_processor(frame_queue, processing_queue, socket_queue)))
        
        # Task gửi dữ liệu WebSocket (nhận từ socket_queue) - chỉ cho OPI
        if "opi" in device_id:
            all_tasks.append(asyncio.create_task(start_sender_worker(socket_queue, ID_CONFIG)))
        
        # Task xử lý Re-ID (nhận từ processing_queue)
        all_tasks.append(asyncio.create_task(start_id(processing_queue, person_reid)))
        
        # Các task giám sát
        all_tasks.append(asyncio.create_task(monitor_queue(frame_queue, "Frame")))
        all_tasks.append(asyncio.create_task(monitor_queue(processing_queue, "Processing")))
        all_tasks.append(asyncio.create_task(monitor_queue(socket_queue, "Socket")))
        all_tasks.append(asyncio.create_task(monitor_system()))
        
        if hasattr(person_reid, '_rabbit_task'):
             all_tasks.append(person_reid._rabbit_task)

        await asyncio.gather(*all_tasks)

    except KeyboardInterrupt:
        logger.info("Đang tắt chương trình một cách an toàn…")
    finally:
        for task in all_tasks:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

        if hasattr(person_reid, 'rabbit'):
            await person_reid.rabbit.stop()
        logger.info("Tất cả tài nguyên đã được dọn dẹp.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy hệ thống ReID với quản lý cơ sở dữ liệu và WebSocket.")
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại cơ sở dữ liệu nếu đã tồn tại.")
    parser.add_argument("--device-id", default="opi", choices=["rpi", "opi"], help="Loại thiết bị (rpi | opi)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Chương trình đã dừng.")