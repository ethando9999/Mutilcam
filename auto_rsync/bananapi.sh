WATCH_DIR="bananapi"  # Đường dẫn đến thư mục cần đồng bộ
DEST="bananapi@192.168.1.186:~/bananapi/"  # Thư mục đích trên Raspberry Pi
# DEST="bananapi@192.168.2.100:~/bananapi/"  # Thư mục đích trên Raspberry Pi

while inotifywait -e close_write -r "$WATCH_DIR"; do
    # Đồng bộ thư mục tới Raspberry Pi
    rsync -avz --exclude="*.pyc" "$WATCH_DIR/" "$DEST"
    echo "Directory synced: $WATCH_DIR"
done