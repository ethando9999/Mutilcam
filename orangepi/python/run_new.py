# file: python/run_new.py (Phiên bản cuối cùng - Tối ưu)

import argparse
import asyncio
import os
import psutil

# Import các module của dự án
from database.create_db import create_database
# from id.main_id_new import start_id, PersonReID # Tạm thời vô hiệu hóa
import config

# Thiết lập logging
from utils.logging_python_orangepi import setup_logging, get_logger
from core.socket_sender import start_sender_worker # Đảm bảo tên file là websocket_sender.py
setup_logging()
logger = get_logger(__name__)


async def monitor_queue(queue: asyncio.Queue, queue_name: str):
    """Log kích thước hàng đợi định kỳ."""
    while True:
        try:
            logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}/{queue.maxsize}")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break

async def monitor_system():
    """Log tải CPU & RAM định kỳ."""
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            logger.info(f"Tải hệ thống - CPU: {cpu_percent:.1f}%, Bộ nhớ: {memory.percent:.1f}%")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break

async def main(args):
    """Điểm vào chính của ứng dụng."""
    logger.info("Khởi động ứng dụng chính…")

    device_id = args.device_id
    logger.info(f"Đang chạy ở chế độ device_id = {device_id}")

    if "opi" in device_id:
        from core.put_RGBD import start_putter
        from core.processing_RGBD import start_processor
        ID_CONFIG = config.OPI_CONFIG
    elif "rpi" in device_id:
        from core.put_frame import start_putter
        from core.processing import start_processor
        ID_CONFIG = config.RPI_CONFIG
    else:
        logger.error(f"device_id không hợp lệ: '{device_id}'. Phải chứa 'rpi' hoặc 'opi'.")
        return

    # Quản lý cơ sở dữ liệu
    db_path = ID_CONFIG.get("db_path", "database.db")
    if args.new_db:
        logger.info(f"Cờ --new-db được bật. Đang tạo lại cơ sở dữ liệu tại '{db_path}'.")
        if os.path.exists(db_path): os.remove(db_path)
        if os.path.exists(f"{db_path}-shm"): os.remove(f"{db_path}-shm")
        if os.path.exists(f"{db_path}-wal"): os.remove(f"{db_path}-wal")
        await create_database(db_path=db_path)
    elif not os.path.exists(db_path):
        logger.warning(f"Cơ sở dữ liệu không tìm thấy. Đang tạo mới tại: {db_path}")
        await create_database(db_path=db_path)
    else:
        logger.info(f"Sử dụng cơ sở dữ liệu hiện có: {db_path}")

    # Khởi tạo các hàng đợi
    frame_queue = asyncio.Queue(maxsize=100)
    processing_queue = asyncio.Queue(maxsize=200)

    all_tasks = []

    try:
        if "opi" in device_id:
            camera_id = ID_CONFIG.get("rgb_camera_id", 0)
            all_tasks.append(asyncio.create_task(start_putter(frame_queue, camera_id=camera_id)))

            calib_path = ID_CONFIG.get("calib_file_path")
            if not calib_path or not os.path.exists(calib_path):
                raise FileNotFoundError(f"LỖI: File hiệu chỉnh không được tìm thấy tại '{calib_path}'.")

            processor_task = asyncio.create_task(
                start_processor(frame_queue, processing_queue, calib_path)
            )
            all_tasks.append(processor_task)

            # Khởi tạo worker gửi dữ liệu WebSocket đa luồng
            logger.info("Phát hiện chế độ 'opi', khởi động WebSocket sender worker...")
            sender_task = asyncio.create_task(
                start_sender_worker(processing_queue, ID_CONFIG)
            )
            all_tasks.append(sender_task)

        else: # Chế độ RPi
            camera_id = ID_CONFIG.get("camera_indices", [0])[0]
            all_tasks.append(asyncio.create_task(start_putter(frame_queue, camera_id=camera_id)))
            processor_task = asyncio.create_task(start_processor(frame_queue, processing_queue))
            all_tasks.append(processor_task)

        # Khởi tạo các tác vụ giám sát
        all_tasks.append(asyncio.create_task(monitor_queue(frame_queue, "Frame")))
        all_tasks.append(asyncio.create_task(monitor_system()))

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
    parser = argparse.ArgumentParser(
        description="Chạy hệ thống ReID với quản lý cơ sở dữ liệu và cấu hình theo thiết bị."
    )
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại cơ sở dữ liệu.")
    parser.add_argument("--device-id", type=str, required=True, help="ID của thiết bị (chứa 'rpi' hoặc 'opi').")

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Chương trình đã dừng bởi người dùng.") 