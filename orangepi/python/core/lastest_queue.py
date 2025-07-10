import asyncio

class LatestFrameQueue:
    """
    Một hàng đợi giống LIFO, chỉ giữ lại mục mới nhất được đưa vào.
    An toàn khi sử dụng với nhiều producer và consumer trong asyncio.
    """
    def __init__(self):
        self._queue = asyncio.Queue(maxsize=1)
        self._lock = asyncio.Lock()

    async def put(self, item):
        # Sử dụng lock để đảm bảo thao tác dọn dẹp và put là nguyên tử
        async with self._lock:
            # Nếu hàng đợi đã có phần tử, lấy ra để loại bỏ nó
            if not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    # An toàn trong trường hợp có consumer khác vừa lấy
                    pass
            # Đặt phần tử mới vào hàng đợi
            await self._queue.put(item)

    async def get(self):
        # Lấy phần tử mới nhất
        return await self._queue.get()