import asyncio
from logging_python_bananapi import get_logger

# Thiết lập logging
logger = get_logger(__name__)

async def start_putter(frame_queue: asyncio.Queue):
    """
    Khởi động putter để đẩy frame vào queue.
    
    Args:
        frame_queue: Queue chứa các frame
    """
    logger.info("Starting putter...")
        
    try:
        await put_frame_queue(frame_queue)
    except asyncio.CancelledError:
        logger.info("Putter task was cancelled.")
    except Exception as e:
        logger.error(f"Error in putter task: {e}")

async def put_frame_queue(frame_queue: asyncio.Queue):
    """
    Đẩy frame vào queue dưới dạng async.
    """
    filename = "python/file.jpg"

    # Đọc file dưới dạng byte
    try:
        with open(filename, "rb") as file:
            byte_content = file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return

    while True:
        if frame_queue.full():
            logger.warning("Frame queue is full. Pausing process frame...")
            await asyncio.sleep(0.1)
            continue
        try:
            # Đẩy 30 frame vào hàng đợi
            for i in range(30):
                await frame_queue.put(byte_content) 
                # logging.info("Frame put success!")
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            await asyncio.sleep(1)