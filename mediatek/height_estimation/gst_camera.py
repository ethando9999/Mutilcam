import asyncio
import os

async def run_command(cmd):
    process = await asyncio.create_subprocess_shell(
        cmd,
        executable="/bin/bash",  # Ép dùng bash để hỗ trợ cú pháp mảng
        env=os.environ.copy(),   # Truyền biến môi trường hiện tại (bao gồm VIDEO_DEV)
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if stdout:
        print(f"[stdout]\n{stdout.decode()}")
    if stderr:
        print(f"[stderr]\n{stderr.decode()}")

async def main():
    # Kiểm tra biến môi trường VIDEO_DEV
    if 'VIDEO_DEV' not in os.environ:
        print("Biến môi trường VIDEO_DEV chưa được thiết lập. Vui lòng thiết lập biến môi trường này.")
        return

    # Sử dụng dấu nháy đơn cho chuỗi caps để tránh nhầm lẫn khi phân tích cú pháp
    cmd1 = (
        "gst-launch-1.0 v4l2src device=${VIDEO_DEV[1]} ! "
        "'image/jpeg,width=640,height=480,format=JPEG' ! jpegdec ! waylandsink sync=false"
    )
    cmd2 = (
        "gst-launch-1.0 v4l2src device=${VIDEO_DEV[3]} ! "
        "'image/jpeg,width=640,height=480,format=JPEG' ! jpegdec ! waylandsink sync=false"
    )

    await asyncio.gather(
        run_command(cmd1),
        run_command(cmd2)
    )

if __name__ == '__main__':
    asyncio.run(main())
