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
        logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}/{queue.maxsize}")
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

    # --------------------------------------------------
    # Chọn cấu hình & module phù hợp theo device_id
    # --------------------------------------------------
    device_id = args.device_id
    if "rpi" in device_id:
        from core.put_frame import start_putter
        from core.processing import start_processor
        from config import RPI_CONFIG as ID_CONFIG
    elif "opi" in device_id:
        from core.put_RGBD import start_putter
        from core.processing_RGBD import start_processor
        from config import OPI_CONFIG as ID_CONFIG
    else:
        raise ValueError("device_id phải là 'rpi' hoặc 'opi'")

    logger.info(f"Đang chạy ở chế độ device_id = {args.device_id}")

    # Lấy đường dẫn cơ sở dữ liệu từ cấu hình
    db_path = ID_CONFIG.get("db_path", "database.db")
    print("db_path: ", db_path)

    # Kiểm tra và quản lý cơ sở dữ liệu với cờ --new-db
    if args.new_db:
        for suffix in ["", "-shm", "-wal"]:
            file = f"{db_path}{suffix}"
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Đã xóa: {file}")
        await create_database(db_path=db_path)
        logger.info(f"Đã tạo cơ sở dữ liệu mới: {db_path}")
    else:
        if not os.path.exists(db_path):
            await create_database(db_path)
            logger.info(f"Cơ sở dữ liệu không được tìm thấy, đã tạo cơ sở dữ liệu mới: {db_path}")
        else:
            logger.info(f"Sử dụng cơ sở dữ liệu hiện có: {db_path}")

    # Khởi tạo các hàng đợi asyncio
    frame_queue = asyncio.Queue(maxsize=10)
    processing_queue = asyncio.Queue(maxsize=1000)

    # --- CẬP NHẬT PHẦN KHỞI TẠO PersonReID ---
    # Định nghĩa các khóa hợp lệ mà PersonReID.__init__ chấp nhận để tránh TypeError
    person_reid_valid_keys = [
        "output_dir", "feature_threshold", "color_threshold", "avg_threshold",
        "top_k", "thigh_weight", "torso_weight", "feature_weight", "color_weight",
        "temp_timeout", "min_detections", "merge_threshold", "face_threshold"
    ]

    # Lọc cấu hình từ ID_CONFIG, chỉ lấy các khóa hợp lệ
    filtered_config = {
        key: ID_CONFIG[key] for key in person_reid_valid_keys if key in ID_CONFIG
    }
    
    # Thêm các tham số được xác định lúc chạy (runtime) hoặc đã được xử lý riêng
    filtered_config["device_id"] = device_id
    filtered_config["db_path"] = db_path  # Sử dụng đường dẫn DB đã được xác định ở trên

    # Khởi tạo PersonReID với các tham số đã được lọc
    person_reid = PersonReID(**filtered_config)
    # --- KẾT THÚC CẬP NHẬT ---

    # Danh sách task để theo dõi và huỷ đúng cách
    all_tasks = []
    person_reid_instance = None # Để kiểm tra trước khi truy cập trong finally
    camera_id = ID_CONFIG.get("rgb_camera_id", 0)

    try:
        person_reid_instance = person_reid
        all_tasks.append(asyncio.create_task(start_putter(frame_queue, camera_id)))
        all_tasks.append(asyncio.create_task(start_processor(frame_queue, processing_queue)))
        all_tasks.append(asyncio.create_task(start_id(processing_queue, person_reid_instance)))
        all_tasks.append(asyncio.create_task(monitor_queue(frame_queue, "Frame")))
        all_tasks.append(asyncio.create_task(monitor_system()))
        
        # Thêm task rabbitmq từ đối tượng person_reid vào danh sách quản lý
        if hasattr(person_reid_instance, '_rabbit_task'):
             all_tasks.append(person_reid_instance._rabbit_task)

        await asyncio.gather(*all_tasks)

    except KeyboardInterrupt:
        logger.info("Đang tắt chương trình một cách an toàn…")
    except asyncio.CancelledError:
        logger.info("Các tác vụ đã bị hủy.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng xảy ra: {str(e)}", exc_info=True)
        raise
    finally:
        # Hủy tất cả task
        for task in all_tasks:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

        if person_reid_instance and hasattr(person_reid_instance, 'rabbit'):
            await person_reid_instance.rabbit.stop()

        # Dọn dẹp hàng đợi
        if not frame_queue.empty():
            logger.info("Đang xóa hàng đợi khung…")
            while not frame_queue.empty():
                frame_queue.get_nowait()
        logger.info("Tất cả tài nguyên đã được dọn dẹp.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chạy hệ thống ReID với quản lý cơ sở dữ liệu.",
    )
    parser.add_argument(
        "--new-db",
        action="store_true",
        help="Xóa và tạo lại cả TempDB và MainDB nếu đã tồn tại.",
    )
    parser.add_argument(
        "--device-id",
        default="rpi",
        help="Loại thiết bị đang chạy ứng dụng (rpi | opi)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Chương trình bị dừng bởi người dùng.")
    except Exception as e:
        logger.error(f"Chương trình dừng do lỗi: {str(e)}")
