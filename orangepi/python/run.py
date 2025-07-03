import argparse
import asyncio
import os
import pika
import psutil
from utils.create_id_db import create_database
from utils.logging_python_orangepi import setup_logging, get_logger
from id.main_id import start_id, PersonReID
from core.put_frame import start_putter
from core.processing import start_processor

# Thiết lập logging
setup_logging()
logger = get_logger(__name__)

async def monitor_queue(queue, queue_name):
    while True:
        logger.info(f"Kích thước hàng đợi {queue_name}: {queue.qsize()}/{queue.maxsize}")
        await asyncio.sleep(5)

async def monitor_system():
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        logger.info(f"Tải hệ thống - CPU: {cpu_percent:.1f}%, Bộ nhớ: {memory.percent:.1f}%")
        await asyncio.sleep(5)

async def init_rabbitmq_queue(queue_name='frame_queue', max_retries=5, retry_delay=5):
    """Khởi tạo hàng đợi RabbitMQ với cơ chế thử lại nếu thất bại."""
    credentials = pika.PlainCredentials('new_user', '123456')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    for attempt in range(max_retries):
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_delete(queue=queue_name)
            logger.info(f"Đã xóa hàng đợi RabbitMQ: {queue_name}")
            channel.queue_declare(queue=queue_name, durable=True)
            connection.close()
            logger.info("Hàng đợi RabbitMQ đã được khởi tạo thành công.")
            return True
        except Exception as e:
            logger.error(f"Không thể khởi tạo hàng đợi RabbitMQ (lần thử {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Không thể khởi tạo hàng đợi RabbitMQ sau nhiều lần thử.")
                return False

async def main(args):
    logger.info("Khởi động ứng dụng chính...")

    # Đường dẫn đến các tệp cơ sở dữ liệu
    temp_db_path = 'temp.db'
    main_db_path = 'database.db'

    # Kiểm tra và quản lý cơ sở dữ liệu với cờ --new-db
    if args.new_db:
        for db_path in [temp_db_path, main_db_path]:
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"Đã xóa cơ sở dữ liệu hiện có: {db_path}")
            await create_database(db_path)
            logger.info(f"Đã tạo cơ sở dữ liệu mới: {db_path}")
    else:
        for db_path in [temp_db_path, main_db_path]:
            if not os.path.exists(db_path):
                await create_database(db_path)
                logger.info(f"Cơ sở dữ liệu không được tìm thấy, đã tạo cơ sở dữ liệu mới: {db_path}")
            else:
                logger.info(f"Sử dụng cơ sở dữ liệu hiện có: {db_path}")

    # Khởi tạo hàng đợi RabbitMQ với thử lại
    if not await init_rabbitmq_queue():
        logger.error("Không thể tiếp tục do lỗi khởi tạo RabbitMQ.")
        return

    # Khởi tạo các hàng đợi asyncio
    frame_queue = asyncio.Queue(maxsize=200)
    processing_queue = asyncio.Queue(maxsize=100)

    # Khởi tạo PersonReID
    person_reid = PersonReID(
        output_dir=args.output_dir,
        feature_threshold=args.feature_threshold,
        color_threshold=args.color_threshold,
        temp_db_path=temp_db_path,
        main_db_path=main_db_path,
        rabbitmq_url="amqp://new_user:123456@localhost/"
    )

    try:
        putter_task = asyncio.create_task(start_putter(frame_queue))
        processor_task = asyncio.create_task(start_processor(frame_queue, processing_queue))
        id_task = asyncio.create_task(start_id(processing_queue, person_reid))
        monitor_task = asyncio.create_task(monitor_queue(frame_queue, "Frame"))
        monitor_system_task = asyncio.create_task(monitor_system())

        await asyncio.gather(putter_task, processor_task, id_task, monitor_task, monitor_system_task)
    except KeyboardInterrupt:
        logger.info("Đang tắt chương trình một cách an toàn...")
    except asyncio.CancelledError:
        logger.info("Các tác vụ đã bị hủy.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng xảy ra: {str(e)}")
        raise
    finally:
        putter_task.cancel()
        processor_task.cancel()
        id_task.cancel()
        monitor_task.cancel()
        monitor_system_task.cancel()
        await asyncio.gather(putter_task, processor_task, id_task, monitor_task, monitor_system_task, return_exceptions=True)
        if not frame_queue.empty():
            logger.info("Đang xóa hàng đợi khung...")
            while not frame_queue.empty():
                frame_queue.get_nowait()
        logger.info("Tất cả tài nguyên đã được dọn dẹp.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy hệ thống ReID với quản lý cơ sở dữ liệu.")
    parser.add_argument('--new-db', action='store_true', help='Xóa và tạo lại cả TempDB và MainDB nếu đã tồn tại.')
    parser.add_argument('--output_dir', type=str, default='output_frames_id', help='Thư mục lưu ảnh crop')
    parser.add_argument('--feature_threshold', type=float, default=0.7, help='Ngưỡng đặc trưng')
    parser.add_argument('--color_threshold', type=float, default=0.5, help='Ngưỡng màu sắc')
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Chương trình bị dừng bởi người dùng.")
    except Exception as e:
        logger.error(f"Chương trình dừng do lỗi: {str(e)}")