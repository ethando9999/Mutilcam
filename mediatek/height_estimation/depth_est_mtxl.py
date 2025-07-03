import cv2
import numpy as np

# Tải tham số hiệu chuẩn
calib_data = np.load('calibration_data.npz')
mtx_l = calib_data['mtx_l']
dist_l = calib_data['dist_l']
mtx_r = calib_data['mtx_r']
dist_r = calib_data['dist_r']
R1 = calib_data['R1']
R2 = calib_data['R2']
P1 = calib_data['P1']
P2 = calib_data['P2']
Q = calib_data['Q']
T= calib_data['T']

# Kích thước ảnh
img_shape = (640, 480)

# Tạo bản đồ chỉnh sửa
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)

# Khởi tạo StereoSGBM
minDisparity = 0
numDisparities = 64
blockSize = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=8 * 3 * blockSize ** 2,
    P2=32 * 3 * blockSize ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Thông số từ bài báo
baseline = np.linalg.norm(T)*0.1  # mm
focal_length = P2[0, 0]  # Tiêu cự (pixel)
print("baseline: ", baseline)
print("focal_length", focal_length)
print("P2", P2[0, 0])
print("P1", P1[0, 0])

frame_l = cv2.imread('calibration/cam1/frame_696.jpg') 
frame_r = cv2.imread('calibration/cam2/frame_705.jpg')


    # Chuyển sang ảnh xám
gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

# Chỉnh sửa hình ảnh
rectified_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
rectified_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)

# Tính bản đồ disparity
disparity = stereo.compute(rectified_l, rectified_r).astype(np.float32) / 16.0
print("disparity", disparity)

# Tính độ sâu
depth = (focal_length * baseline) / (disparity + 1e-6)  # Thêm 1e-6 để tránh chia cho 0
print("depth: ", depth)

# Chuẩn hóa để hiển thị
disparity_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth_display = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

