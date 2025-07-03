import cv2
from utils.config_camera import CameraHandler

cam0 = CameraHandler(camera_index=0)

frame = cam0.capture_main_frame()

cv2.imwrite('captured_frame.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print("Đã lưu captured_frame.jpg")

cam0.stop_camera()