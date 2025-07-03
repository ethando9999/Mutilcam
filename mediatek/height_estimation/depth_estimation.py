import cv2
import numpy as np
import glob
import re
import os

def natural_keys(text):
    """
    Hàm sắp xếp theo thứ tự tự nhiên (natural sort)
    """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

# Cấu hình mẫu bàn cờ
pattern_size = (9, 6)    # Số góc nội (9 cột, 6 hàng)
square_size = 0.028      # Mỗi ô bàn cờ có kích thước 2.8 cm = 0.028 m

# Tạo objp là các tọa độ (x,y,z) thật, với z=0
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size  # Nhân với kích thước thật của mỗi ô (đơn vị: mét)

# Các biến lưu điểm 3D và 2D
objpoints = []
imgpoints_left = []
imgpoints_right = []

# Đọc danh sách ảnh cho camera trái và phải
images_left = sorted(glob.glob('calibration3/cam1/*.jpg'), key=natural_keys)
images_right = sorted(glob.glob('calibration3/cam2/*.jpg'), key=natural_keys)

for left_path, right_path in zip(images_left, images_right):
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    
    # Chuyển sang ảnh xám
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Tìm góc bàn cờ trên ảnh trái và phải
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)
    
    if ret_left and ret_right:
        # Lưu objp
        objpoints.append(objp)
        
        # Tinh chỉnh vị trí góc
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
        
        imgpoints_left.append(corners2_left)
        imgpoints_right.append(corners2_right)

# Lấy kích thước ảnh (width, height) từ ảnh cuối
h, w = gray_left.shape[:2]
img_size = (w, h)

# Hiệu chuẩn camera trái
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_size, None, None
)

# Hiệu chuẩn camera phải
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_size, None, None
)

# Tiêu chí tối ưu stereo
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
flags = cv2.CALIB_FIX_INTRINSIC  # Giữ nguyên tham số nội tại đã có

# StereoCalibrate
ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left, imgpoints_right,
    mtx_left, dist_left,
    mtx_right, dist_right,
    img_size,
    criteria=criteria_stereo,
    flags=flags
)

# Tính các ma trận Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left,
    mtx_right, dist_right,
    img_size,
    R, T,
    alpha=0
)

# Tạo bản đồ chỉnh sửa (rectification map)
map1_left, map2_left = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1,
    img_size, cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2,
    img_size, cv2.CV_16SC2
)

# Đọc cặp ảnh thực tế
img_test_left = cv2.imread('calibration3/cam1/frame_516.jpg')
img_test_right = cv2.imread('calibration3/cam2/frame_516.jpg')

# Chuyển sang ảnh xám
gray_test_left = cv2.cvtColor(img_test_left, cv2.COLOR_BGR2GRAY)
gray_test_right = cv2.cvtColor(img_test_right, cv2.COLOR_BGR2GRAY)

# Rectify hai ảnh
rect_left = cv2.remap(gray_test_left, map1_left, map2_left, cv2.INTER_LINEAR)
rect_right = cv2.remap(gray_test_right, map1_right, map2_right, cv2.INTER_LINEAR)

# Thiết lập thông số cho StereoSGBM (hoặc StereoBM)
window_size = 7
min_disp = 0
num_disp = 16 * 5  # Phải là bội số của 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8*3*window_size**2,
    P2=32*3*window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

disparity = stereo.compute(rect_left, rect_right).astype(np.float32) / 16.0

# Hiển thị thử bản đồ disparity
disparity_show = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_show = np.uint8(disparity_show)
# cv2.imshow("Disparity", disparity_show)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Lấy tiêu cự và baseline
focal_length = P1[0, 0]     # pixel
baseline = np.linalg.norm(T)  # mét
# baseline = 0.028

# Tạo depth map
depth_map = np.zeros(disparity.shape, np.float32)
# Chỉ tính nơi disparity > 0
mask = disparity > 0
depth_map[mask] = (focal_length * baseline) / disparity[mask]

print("Baseline (m): ", baseline)
print("Focal length (px): ", focal_length)
print("Disparity: ", disparity)

# Xem một vài giá trị depth
print("Depth sample: ", depth_map[mask][:50])  # in thử 50 giá trị đầu

# Hiển thị Depth map
depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = np.uint8(depth_norm)
# cv2.imshow("Depth Map", depth_norm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
