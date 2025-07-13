# file: python/run_new.py (Phiên bản cuối cùng - Đã sửa lỗi)

import argparse
import asyncio
import os
import psutil

# Import các module của dự án
from database.create_db import create_database
from id.main_id_new import start_id, PersonReID
import config
from core.socket_sender import start_socket_sender

# Thiết lập logging
from utils.logging_python_orangepi import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

async def monitor_queue(queue: asyncio.Queue, queue_name: str):
    """Log kích thước hàng đợi định kỳ."""
    while True:
        try:
            max_size_str = f"/{queue.maxsize}" if queue.maxsize > 0 else ""
            logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}{max_size_str}")
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
    else:
        logger.error(f"device_id không hợp lệ: '{device_id}'.")
        return

    # Quản lý cơ sở dữ liệu
    db_path = ID_CONFIG.get("db_path")
    if args.new_db:
        logger.info(f"Cờ --new-db được bật. Đang tạo lại cơ sở dữ liệu tại '{db_path}'.")
        for suffix in ["", "-shm", "-wal"]:
            if os.path.exists(f"{db_path}{suffix}"): os.remove(f"{db_path}{suffix}")
        await create_database(db_path=db_path)
    elif not os.path.exists(db_path):
        await create_database(db_path=db_path)

    # Khởi tạo các hàng đợi (sử dụng maxsize từ snippet của bạn)
    frame_queue = asyncio.Queue(maxsize=5)
    processing_queue = asyncio.Queue(maxsize=200)
    people_count_queue = asyncio.Queue(maxsize=1)
    height_queue = asyncio.Queue(maxsize=1)
    id_socket_queue = asyncio.Queue()

    person_reid = None
    all_tasks = []

    try:
        # --- KHỞI TẠO PersonReID MỘT CÁCH AN TOÀN VỚI filtered_config ---
        person_reid_valid_keys = [
            "output_dir", "feature_threshold", "color_threshold", "avg_threshold",
            "top_k", "thigh_weight", "torso_weight", "feature_weight", "color_weight",
            "temp_timeout", "min_detections", "merge_threshold", "face_threshold"
        ]
        
        # Lọc cấu hình, chỉ lấy các khóa hợp lệ cho PersonReID.__init__
        filtered_config = {key: ID_CONFIG[key] for key in person_reid_valid_keys if key in ID_CONFIG}
        
        # Thêm các tham số runtime cần thiết
        filtered_config["device_id"] = device_id
        filtered_config["db_path"] = db_path
        filtered_config["id_socket_queue"] = id_socket_queue
        
        # Khởi tạo person_reid với config đã được lọc an toàn
        person_reid = PersonReID(**filtered_config)

        # --- KHỞI TẠO CÁC TÁC VỤ CHÍNH ---

        # 1. Tác vụ Putter
        camera_id = ID_CONFIG.get("rgb_camera_id", 0)
        all_tasks.append(asyncio.create_task(start_putter(frame_queue, camera_id=camera_id)))

        # 2. Tác vụ Processor
        calib_path = ID_CONFIG.get("calib_file_path")
        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError(f"LỖI: File hiệu chỉnh không được tìm thấy tại '{calib_path}'.")
        
        all_tasks.append(asyncio.create_task(start_processor(
            frame_queue, processing_queue, people_count_queue, height_queue, calib_path
        )))
        
        # 3. Tác vụ ReID (GỌI VỚI 2 THAM SỐ)
        # Để tương thích với file main_id_new.py hiện tại của bạn
        all_tasks.append(asyncio.create_task(start_id(
            processing_queue,
            person_reid
        )))

        # 4. Tác vụ WebSocket Senders
        all_tasks.append(asyncio.create_task(start_socket_sender(people_count_queue, ID_CONFIG["SOCKET_COUNT_URI"])))
        all_tasks.append(asyncio.create_task(start_socket_sender(height_queue, ID_CONFIG["SOCKET_HEIGHT_URI"])))
        all_tasks.append(asyncio.create_task(start_socket_sender(id_socket_queue, ID_CONFIG["SOCKET_ID_URI"])))
        
        # 5. Tác vụ giám sát
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
            if not task.done(): task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        if person_reid and hasattr(person_reid, 'rabbit') and person_reid.rabbit:
            await person_reid.rabbit.stop()
            
        logger.info("Tất cả tài nguyên đã được dọn dẹp. Chương trình kết thúc.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy hệ thống ReID.")
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại cơ sở dữ liệu.")
    parser.add_argument("--device-id", type=str, default="opi", help="ID của thiết bị (chứa 'opi').")
    
    args = parser.parse_args()
    asyncio.run(main(args))