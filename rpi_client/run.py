import asyncio
from core.put_frame import start_putter
from send_frame import start_sender
from core.processing import start_processor
# from put_frame import start_putter
from utils.camera_log import setup_logging, get_logger

# Thiết lập logger khi khởi động
setup_logging()
logger = get_logger(__name__)

async def main():
    """
    Hàm main chạy đồng thời putter, processor và sender với shared queues. 
    Putter đẩy frame test vào frame_queue.
    Processor xử lý frame từ frame_queue và đưa vào processed_frame_queue.
    Sender lấy frame từ processed_frame_queue để gửi đi.
    """
    logger.info("Starting main application...")
    
    # Khởi tạo các queue
    frame_queue = asyncio.Queue(maxsize=10000)
    processed_frame_queue = asyncio.Queue(maxsize=10000)

    try: 

        putter_task = asyncio.create_task(start_putter(frame_queue))
        # await asyncio.sleep(1)  # Chờ server khởi động 

        # Tạo task cho processor và sender 
        processor_task = asyncio.create_task(
            start_processor(
                frame_queue=frame_queue,
                processed_frame_queue=processed_frame_queue,
                num_workers=1 # Số lượng processor workers
            )
        )

        # sender_task = asyncio.create_task( 
        #     start_sender(
        #         frame_queue=processed_frame_queue,
        #         include_put_frame=False,
        #         num_workers=1
        #     )
        # )

        # Chạy đồng thời tất cả tasks 
        await asyncio.gather(
            putter_task,
            processor_task, 
            # sender_task
        )

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except asyncio.CancelledError:
        logger.info("Tasks were cancelled.")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Hủy tất cả tasks
        putter_task.cancel()
        processor_task.cancel()
        sender_task.cancel()
        
        # Chờ tasks kết thúc
        await asyncio.gather(
            putter_task, 
            processor_task, 
            sender_task, 
            return_exceptions=True
        )
        logger.info("All tasks have been stopped.") 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program stopped by user.") 
