import asyncio
from core.put_frame import start_putter
from send_packets.send_manager import start_send_manager
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
    human_queue = asyncio.Queue(maxsize=1000)
    head_queue = asyncio.Queue(maxsize=1000)
    right_arm_queue = asyncio.Queue(maxsize=1000)
    left_arm_queue = asyncio.Queue(maxsize=1000)

    try: 

        putter_task = asyncio.create_task(start_putter(frame_queue))
        # await asyncio.sleep(1)  # Chờ server khởi động

        # Tạo task cho processor và sender 
        processor_task = asyncio.create_task(
            start_processor(
                frame_queue=frame_queue,
                human_queue=human_queue,
                head_queue=head_queue,
                right_arm_queue=right_arm_queue,
                left_arm_queue=left_arm_queue,
            )
        )

        sender_task = asyncio.create_task( 
            start_send_manager(
                human_queue=human_queue,
                head_queue=head_queue,
                right_arm_queue=right_arm_queue,
                left_arm_queue=left_arm_queue
            )
        )

        # Chạy đồng thời tất cả tasks  
        await asyncio.gather(
            putter_task,
            processor_task, 
            sender_task
        )
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
