# file: python/run_new.py (Tối ưu sử dụng ProcessID thay cho start_id/PersonReID)

import argparse
import asyncio
import os
import psutil

# Import các module của dự án
from database.create_db import create_database
from id.process_id import ProcessID
import config
from core.socket_sender import start_socket_sender

# Thiết lập logging
from utils.logging_python_orangepi import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

async def monitor_queue(queue: asyncio.Queue, queue_name: str):
    while True:
        try:
            max_size_str = f"/{queue.maxsize}" if queue.maxsize > 0 else ""
            logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}{max_size_str}")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break

async def monitor_system():
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
        from core.processing_RGBD import start_processor
        ID_CONFIG = config.OPI_CONFIG
    else:
        logger.error(f"device_id không hợp lệ: '{device_id}'.") 
        return

    db_path = ID_CONFIG.get("db_path")
    if args.new_db:
        logger.info(f"Cờ --new-db được bật. Đang tạo lại cơ sở dữ liệu tại '{db_path}'.")
        for suffix in ["", "-shm", "-wal"]:
            if os.path.exists(f"{db_path}{suffix}"): os.remove(f"{db_path}{suffix}")
        await create_database(db_path=db_path)
    elif not os.path.exists(db_path):
        await create_database(db_path=db_path)

    frame_queue = asyncio.Queue(maxsize=5)
    processing_queue = asyncio.Queue(maxsize=200)
    people_count_queue = asyncio.Queue(maxsize=1)
    height_queue = asyncio.Queue(maxsize=1)
    id_socket_queue = asyncio.Queue()

    all_tasks = []

    try:
        # Khởi tạo process handler
        processor = ProcessID({
            **ID_CONFIG,
            "device_id": device_id,
            "db_path": db_path
        },
        processing_queue,
        id_socket_queue
        )

        # 1. Putter
        camera_id = ID_CONFIG.get("rgb_camera_id", 0)
        all_tasks.append(asyncio.create_task(start_putter(frame_queue, camera_id=camera_id)))

        # 2. Processor
        calib_path = ID_CONFIG.get("calib_file_path")
        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError(f"LỖI: File hiệu chỉnh không được tìm thấy tại '{calib_path}'.")

        all_tasks.append(asyncio.create_task(start_processor(
            frame_queue, processing_queue, people_count_queue, height_queue, calib_path
        )))

        # 3. ID xử lý
        all_tasks.append(asyncio.create_task(processor.run()))

        # 4. WebSocket
        all_tasks.append(asyncio.create_task(start_socket_sender(people_count_queue, ID_CONFIG["SOCKET_COUNT_URI"])))
        all_tasks.append(asyncio.create_task(start_socket_sender(height_queue, ID_CONFIG["SOCKET_HEIGHT_URI"])))
        all_tasks.append(asyncio.create_task(start_socket_sender(id_socket_queue, ID_CONFIG["SOCKET_ID_URI"])))

        # 5. Giám sát
        all_tasks.append(asyncio.create_task(monitor_queue(frame_queue, "Frame")))
        all_tasks.append(asyncio.create_task(monitor_queue(processing_queue, "Processing")))
        all_tasks.append(asyncio.create_task(monitor_queue(id_socket_queue, "ID Socket")))

        logger.info(f"Đã khởi tạo {len(all_tasks)} tác vụ. Bắt đầu vòng lặp chính.")
        await asyncio.gather(*all_tasks)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Phát hiện tín hiệu dừng. Đang tắt chương trình...")
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
    parser = argparse.ArgumentParser(description="Chạy hệ thống ReID.")
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại cơ sở dữ liệu.")
    parser.add_argument("--device-id", type=str, default="opi", help="ID của thiết bị (chứa 'opi').")
    args = parser.parse_args()
    asyncio.run(main(args))
