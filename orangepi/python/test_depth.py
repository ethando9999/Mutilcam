import os
import cv2
import numpy as np

depth_path = "your_depth_file.npy"  # <-- thay bằng path thực tế
if depth_path and os.path.exists(depth_path):
    # Đọc lại depth frame mỗi vòng lặp (nếu bạn ghi file liên tục)
    depth = np.load(depth_path)

while True:
    # Chuyển depth sang ảnh 8-bit để hiển thị
    tof_color = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    tof_color_map = cv2.applyColorMap(tof_color, cv2.COLORMAP_JET)

    # Hiển thị ảnh
    cv2.imshow("TOF Depth Colormap", tof_color_map)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # Bấm ESC để thoát
        break

cv2.destroyAllWindows()
