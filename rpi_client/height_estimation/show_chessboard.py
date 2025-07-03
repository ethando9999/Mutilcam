import cv2
import numpy as np
import glob
import re

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order.
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

# Định nghĩa kích thước bàn cờ (số góc trong mỗi hàng, số góc trong mỗi cột)
pattern_size = (9, 6)

# Chuẩn bị các điểm không gian thực: (0,0,0), (1,0,0), ...
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Các mảng lưu trữ điểm 3D và điểm 2D
objpoints = []       # điểm 3D trong không gian thực
imgpoints_left = []  # điểm ảnh của camera trái
imgpoints_right = [] # điểm ảnh của camera phải

# Đường dẫn chứa ảnh hiệu chuẩn (chỉnh sửa đường dẫn cho phù hợp)
images_left = sorted(glob.glob('calibration3/cam1/*.jpg'), key=natural_keys)
images_right = sorted(glob.glob('calibration3/cam2/*.jpg'), key=natural_keys) 

for left_img_path, right_img_path in zip(images_left, images_right):
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Tìm góc bàn cờ
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        
        # Cải thiện vị trí góc với cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        
        imgpoints_left.append(corners2_left)
        imgpoints_right.append(corners2_right)
        
        # Vẽ các góc đã tìm được lên ảnh
        cv2.drawChessboardCorners(img_left, pattern_size, corners2_left, ret_left)
        cv2.drawChessboardCorners(img_right, pattern_size, corners2_right, ret_right)
        
        # Hiển thị ảnh với các góc đã được đánh dấu
        cv2.imshow('Left Image', img_left)
        cv2.imshow('Right Image', img_right)
        cv2.waitKey(500)  # Chờ 500ms trước khi chuyển sang ảnh tiếp theo

cv2.destroyAllWindows()

# Hiệu chuẩn từng camera riêng biệt
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
