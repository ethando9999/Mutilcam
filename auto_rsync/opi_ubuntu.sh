WATCH_DIR="orangepi"  # Đường dẫn đến thư mục cần đồng bộ
DEST="ubuntu@192.168.1.166:~/orangepi2/"  # Thư mục đích trên OrangePi

while inotifywait -e close_write -r "$WATCH_DIR"; do
    # Đồng bộ thư mục tới OrangePi, bao gồm .env
    rsync -avz --include=".env" --exclude="*.pyc" --exclude=".git/" --exclude="*.log" --exclude="env/" --exclude="venv/" --exclude="ven/" --exclude="rknn_venv/" "$WATCH_DIR/" "$DEST"
    echo "Directory synced: $WATCH_DIR to $DEST at $(date)"
done