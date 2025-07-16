import asyncio
from core.put_RGBD import start_putter

async def main():
    frame_queue = asyncio.Queue(maxsize=1000)
    await start_putter(frame_queue, 0) 

if __name__ == "__main__":
    asyncio.run(main())
