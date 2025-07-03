WATCH_DIR="object_detection"  # Đường dẫn đến thư mục cần đồng bộ
DEST="orangepi@192.168.1.123:~/object_detection/"  # Thư mục đích trên Raspberry Pi

while inotifywait -e close_write -r "$WATCH_DIR"; do
    # Đồng bộ thư mục tới Raspberry Pi
    rsync -avz --exclude="*.pyc" "$WATCH_DIR/" "$DEST"
    echo "Directory synced: $WATCH_DIR"
done