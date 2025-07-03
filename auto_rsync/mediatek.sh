WATCH_DIR="mediatek"  # Đường dẫn đến thư mục cần đồng bộ
DEST="ubuntu@192.168.1.203:~/mediatek/"  # Thư mục đích trên Raspberry Pi

while inotifywait -e close_write -r "$WATCH_DIR"; do
    # Đồng bộ thư mục tới Raspberry Pi
    rsync -avz --exclude="*.pyc" "$WATCH_DIR/" "$DEST"
    echo "Directory synced: $WATCH_DIR"
done