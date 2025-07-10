# file: run_detect.py
import argparse
import asyncio
import os
import psutil

# Import các module ứng dụng
from database.create_db import create_database
from id.main_id_new import start_id, PersonReID
from utils.logging_python_orangepi import setup_logging, get_logger
from core.processing import start_processor
from core.put_frame_opi import start_putter
from orangepi.python.config_x import DEVICE_ID_CONFIG_2 as ID_CONFIG 

setup_logging()
logger = get_logger(__name__)

async def monitor_queue(queue, name):
    while True:
        logger.info(f"Hàng đợi {name}: {queue.qsize()}/{queue.maxsize}")
        await asyncio.sleep(5)

async def monitor_system():
    while True:
        logger.info(f"Tải hệ thống - CPU: {psutil.cpu_percent():.1f}%, Mem: {psutil.virtual_memory().percent:.1f}%")
        await asyncio.sleep(10)

async def main(args):
    logger.info("="*50)
    logger.info("Khởi động ứng dụng Re-ID với luồng xử lý tối ưu...")
    logger.info("="*50)

    # Tách cấu hình cho Putter (phần cứng)
    putter_config = {
        "calib_file_path": ID_CONFIG.get("calib_file_path"),
        "results_dir": ID_CONFIG.get("results_dir", "run_results/"),
        "rgb_camera_index": ID_CONFIG.get("rgb_camera_index", 0),
        "slave_ip": ID_CONFIG.get("slave_ip"),
        "tcp_port": ID_CONFIG.get("tcp_port", 5005)
    }
    if not putter_config["calib_file_path"] or not putter_config["slave_ip"]:
        logger.error("LỖI CẤU HÌNH: 'calib_file_path' hoặc 'slave_ip' chưa được thiết lập trong config.py!")
        return
    logger.info(f"Chế độ: {args.device_id.upper()} | IP Slave: {putter_config['slave_ip']}")

    # ======================== SỬA LỖI TẠI ĐÂY ========================
    # Lọc ra các key không thuộc về PersonReID để tránh TypeError
    hardware_keys_to_exclude = [
        "calib_file_path", "slave_ip", "results_dir",
        "rgb_camera_index", "tcp_port"
    ]
    person_reid_config = {
        key: value for key, value in ID_CONFIG.items()
        if key not in hardware_keys_to_exclude
    }
    # =================================================================

    # Quản lý DB
    db_path = ID_CONFIG.get("db_path", "database.db")
    if args.new_db and os.path.exists(db_path): os.remove(db_path)
    if not os.path.exists(db_path):
        await create_database(db_path=db_path)
        logger.info(f"Đã tạo DB mới: {db_path}")

    # Khởi tạo các thành phần với config đã được lọc
    frame_queue = asyncio.Queue(maxsize=50)
    processing_queue = asyncio.Queue(maxsize=1000)
    person_reid = PersonReID(**person_reid_config) # <<< SỬ DỤNG CONFIG ĐÃ LỌC

    tasks = []
    try:
        tasks.extend([
            asyncio.create_task(start_putter(frame_queue, putter_config)),
            asyncio.create_task(start_processor(frame_queue, processing_queue)),
            asyncio.create_task(start_id(processing_queue, person_reid)),
            asyncio.create_task(monitor_queue(frame_queue, "Frame")),
            asyncio.create_task(monitor_system()),
        ])
        if hasattr(person_reid, '_rabbit_task'): tasks.append(person_reid._rabbit_task)
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Đang tắt chương trình...")
    finally:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if hasattr(person_reid, 'rabbit'): await person_reid.rabbit.stop()
        logger.info("Chương trình kết thúc.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy hệ thống Re-ID với giao tiếp Master-Slave.")
    parser.add_argument("--new-db", action="store_true", help="Xóa và tạo lại DB.")
    parser.add_argument("--device-id", default="opi", choices=["opi"], help="Chỉ hỗ trợ 'opi'.")
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass