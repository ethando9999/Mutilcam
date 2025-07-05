WATCH_DIR="aducam_tof"  # Đường dẫn đến thư mục cần đồng bộ
# DEST="pi003@192.168.1.101:~/pi_client/"  # Thư mục đích trên Raspberry Pi
DEST="pi02@192.168.1.123:~/aducam_tof/"  

while inotifywait -e close_write -r "$WATCH_DIR"; do
    # Đồng bộ thư mục tới Raspberry Pi
    rsync -avz --exclude="*.pyc" "$WATCH_DIR/" "$DEST"
    echo "Directory synced: $WATCH_DIR" 
done    