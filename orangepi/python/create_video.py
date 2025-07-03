import cv2

# Mở camera (0 là mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera")
    exit()

# Lấy thông tin độ phân giải
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = 30  # có thể đổi 30 nếu cần

# Định nghĩa codec và tạo đối tượng VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # hoặc 'MJPG', 'MP4V'
out = cv2.VideoWriter('video_5mp.avi', fourcc, fps, (width, height))

print("Đang quay... Nhấn 'q' để dừng.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)  # Ghi frame vào file

    # Hiển thị để theo dõi
    # cv2.imshow('Recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
