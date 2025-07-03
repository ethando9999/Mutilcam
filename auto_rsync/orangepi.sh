WATCH_DIR="orangepi"  # Đường dẫn đến thư mục cần đồng bộ
DEST="ubuntu@192.168.1.229:~/orangepi/"  # Thư mục đích trên Raspberry Pi

while inotifywait -e close_write -r "$WATCH_DIR"; do
    # Đồng bộ thư mục tới Raspberry Pi
    rsync -avz --exclude="*.pyc" "$WATCH_DIR/" "$DEST"
    echo "Directory synced: $WATCH_DIR"
done