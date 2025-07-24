import cv2

# Mở camera mặc định (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame.")
        break

    # Hiển thị frame
    cv2.imshow('Camera', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
