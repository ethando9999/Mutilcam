# file: python/run_pipeline.py

import argparse
import asyncio
import os
import psutil
from typing import Dict, Any, List

# --- Import các thành phần cốt lõi của kiến trúc ---
from database.create_db import create_database
from id.main_id_new import start_id, PersonReID
# Đảm bảo các đường dẫn import này trỏ đến các file đã được tối ưu
from track_local.tracking_3Dpro import Track3DPro
from core.lastest_queue import LatestFrameQueue # Sửa lỗi chính tả "lastest" -> "latest"
from config import OPI_CONFIG, RPI_CONFIG
from utils.logging_python_orangepi import setup_logging, get_logger

# --- Thiết lập Logging ---
setup_logging()
logger = get_logger(__name__)

# --- Các hàm phụ trợ (Helpers) ---

def _filter_config(source_config: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Lọc và trả về một dictionary mới chỉ chứa các key được yêu cầu."""
    return {key: source_config.get(key) for key in required_keys if key in source_config}

async def monitor_queue(queue: asyncio.Queue, name: str):
    """Giám sát và log kích thước hàng đợi định kỳ."""
    while True:
        try:
            if isinstance(queue, LatestFrameQueue):
                 logger.info(f"Hàng đợi {name}: [LIFO queue active]")
            else:
                max_size = getattr(queue, 'maxsize', 'N/A')
                logger.info(f"Hàng đợi {name}: {queue.qsize()}/{max_size}")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break

async def monitor_system():
    """Giám sát và log tải hệ thống định kỳ."""
    while True:
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            logger.info(f"Tải hệ thống - CPU: {cpu:.1f}%, Bộ nhớ: {mem:.1f}%")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            break


# --- Hàm chính của ứng dụng ---
async def main(args: argparse.Namespace):
    """
    Điểm vào chính, điều phối pipeline theo kiến trúc phân tầng.
    """
    logger.info("="*50)
    logger.info("🚀 Khởi động Pipeline Re-ID Phân Tầng...")
    logger.info("="*50)

    # 1. Chọn Cấu hình và Module
    logger.info(f"Chế độ thiết bị được chọn: {args.device_id.upper()}")
    if "opi" in args.device_id:
        ID_CONFIG = OPI_CONFIG
        from core.put_RGBD_final import start_putter
        from core.processing_RGBD_final import start_processor
    else:
        logger.error(f"Kiến trúc này hiện được tối ưu cho 'opi'.")
        return

    # 2. Quản lý Cơ sở dữ liệu
    db_path = ID_CONFIG.get("db_path")
    if not db_path:
        logger.error("Lỗi cấu hình: 'db_path' không được tìm thấy trong config.")
        return

    if args.new_db:
        logger.warning(f"Cờ --new-db được bật. Đang xóa DB (nếu có): {db_path}")
        for suffix in ["", "-shm", "-wal"]:
            db_file_part = f"{db_path}{suffix}"
            if os.path.exists(db_file_part):
                os.remove(db_file_part)
    
    if not os.path.exists(db_path):
       await create_database(db_path=db_path)

    # 3. Lọc cấu hình và Khởi tạo các Thành phần
    logger.info("Phân tách cấu hình và khởi tạo các thành phần...")
    
    # Định nghĩa tường minh các key mà mỗi thành phần cần
    PUTTER_KEYS = ["rgb_camera_id", "slave_ip", "tcp_port", "rgb_framerate", "bg_learning_time"]
    TRACKER_KEYS = ["calib_file_path", "max_time_lost"]
    PROCESSOR_KEYS = ["calib_file_path", "device_id", "model_path", "results_dir", "distance_threshold_m"]
    PERSON_REID_KEYS = [
        "device_id", "db_path", "output_dir", "feature_threshold", "color_threshold",
        "avg_threshold", "top_k", "thigh_weight", "torso_weight", "feature_weight",
        "color_weight", "temp_timeout", "min_detections", "merge_threshold",
        "face_threshold"
    ]

    putter_config = _filter_config(ID_CONFIG, PUTTER_KEYS)
    tracker_config = _filter_config(ID_CONFIG, TRACKER_KEYS)
    processor_config = _filter_config(ID_CONFIG, PROCESSOR_KEYS)
    person_reid_config = _filter_config(ID_CONFIG, PERSON_REID_KEYS)

    frame_queue = LatestFrameQueue()
    final_result_queue = asyncio.Queue(maxsize=200)

    if not tracker_config.get("calib_file_path"):
        logger.error("Lỗi cấu hình: 'calib_file_path' là bắt buộc cho Tracker.")
        return
        
    tracker = Track3DPro(**tracker_config)
    person_reid = PersonReID(**person_reid_config)

    # 4. Quản lý và Chạy Tác vụ
    all_tasks = []
    try:
        logger.info("Đang tạo và khởi chạy các tác vụ của pipeline...")
        all_tasks = [
            asyncio.create_task(start_putter(frame_queue, putter_config)),
            asyncio.create_task(start_processor(frame_queue, final_result_queue, tracker, processor_config)),
            asyncio.create_task(start_id(final_result_queue, person_reid)),
            asyncio.create_task(monitor_queue(frame_queue, "Frame (LIFO)")),
            asyncio.create_task(monitor_queue(final_result_queue, "Final Result")),
            asyncio.create_task(monitor_system()),
        ]
        if hasattr(person_reid, '_rabbit_task'):
            all_tasks.append(person_reid._rabbit_task)

        logger.info(f"✅ Đã khởi chạy {len(all_tasks)} tác vụ. Pipeline đang hoạt động.")
        await asyncio.gather(*all_tasks)

    except KeyboardInterrupt:
        logger.warning("Phát hiện tín hiệu dừng...")
    finally:
        logger.info("Bắt đầu quá trình dọn dẹp và tắt chương trình...")
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        if hasattr(person_reid, 'rabbit') and hasattr(person_reid.rabbit, 'stop'):
            await person_reid.rabbit.stop()
        logger.info("Chương trình đã kết thúc một cách an toàn.")


# --- Điểm vào của Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Pipeline Re-ID Phân Tầng.")
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại DB.")
    parser.add_argument("--device-id", default="opi", choices=["opi", "rpi"], help="Chế độ thiết bị.")
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Chương trình đã dừng bởi người dùng.")
