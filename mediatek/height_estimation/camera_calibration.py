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

def calibrate_stereo_camera(folder_left, folder_right, pattern_size=(9, 6), square_size=2.8, real_baseline=5.5, step=10):
    """
    Hàm hiệu chuẩn stereo (hai camera trái và phải) dựa trên ảnh bàn cờ.

    Parameters:
      - folder_left: Thư mục chứa ảnh của camera trái (ví dụ: 'calibration4/cam1')
      - folder_right: Thư mục chứa ảnh của camera phải (ví dụ: 'calibration4/cam2')
      - pattern_size: Kích thước bàn cờ (số góc nội theo cột, hàng), mặc định (9, 6)
      - square_size: Kích thước mỗi ô bàn cờ (đơn vị cm), mặc định 2.8
      - real_baseline: Baseline thực (đơn vị cm), mặc định 5.5
      - step: Bước nhảy lấy mẫu từ danh sách ảnh, mặc định 10

    Hàm sẽ lưu các tham số hiệu chuẩn vào file 'calibration_data.npz'
    và in ra kết quả tiêu cự và baseline đã hiệu chỉnh.
    """
    # Chuẩn bị tọa độ 3D của bàn cờ (z = 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Các list lưu trữ điểm 3D và 2D của ảnh bàn cờ
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # Đọc danh sách ảnh của camera trái và phải và sắp xếp theo thứ tự tự nhiên
    images_left = sorted(glob.glob(os.path.join(folder_left, '*.jpg')), key=natural_keys)
    images_right = sorted(glob.glob(os.path.join(folder_right, '*.jpg')), key=natural_keys)

    # Áp dụng bước nhảy lấy mẫu
    images_left_sampled = images_left[::step]
    images_right_sampled = images_right[::step]

    # Tiêu chí dừng của thuật toán tìm subPixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for left_path, right_path in zip(images_left_sampled, images_right_sampled):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

        if ret_left and ret_right:
            print(f"Tìm thấy bàn cờ cho cam trái: {os.path.basename(left_path)} và cam phải: {os.path.basename(right_path)}")
            objpoints.append(objp)

            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            imgpoints_left.append(corners2_left)
            imgpoints_right.append(corners2_right)

    if len(objpoints) == 0:
        print("Không tìm thấy bất kỳ mẫu bàn cờ nào, không thể hiệu chuẩn!")
        return

    # Lấy kích thước ảnh từ ảnh cuối cùng
    h, w = gray_left.shape[:2]
    img_size = (w, h)

    # Hiệu chuẩn riêng cho từng camera
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None
    )

    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None
    )

    # Thực hiện hiệu chuẩn stereo, cố định intrinsic đã hiệu chuẩn
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC

    ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size,
        criteria=criteria_stereo,
        flags=flags
    )

    # Tính toán ma trận Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_left, dist_left,
        mtx_right, dist_right,
        img_size,
        R, T,
        alpha=0
    )

    # Tạo bản đồ hiệu chỉnh
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1,
        img_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2,
        img_size, cv2.CV_16SC2
    )

    # Lấy tiêu cự và baseline từ ma trận P1 và vector T (hiệu chuẩn)
    focal_length_calib = P1[0, 0]      # Tiêu cự theo pixel
    baseline_calib = np.linalg.norm(T)  # Baseline tính được (cm)

    print("Baseline (cm) hiệu chuẩn: ", baseline_calib)
    print("Tiêu cự (px) theo hiệu chuẩn: ", focal_length_calib)

    # Điều chỉnh tiêu cự nếu baseline thực tế khác baseline đã hiệu chuẩn
    scale = real_baseline / baseline_calib
    focal_length_corrected = focal_length_calib * scale

    print("Hệ số scale: ", scale)
    print("Tiêu cự (px) đã hiệu chỉnh: ", focal_length_corrected)

    # Lưu các tham số hiệu chuẩn vào file
    file_name = 'calibration_data1.npz'
    np.savez(file_name, 
             mtx_l=mtx_left, dist_l=dist_left, 
             mtx_r=mtx_right, dist_r=dist_right, 
             R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, 
             scale=scale, focal_length_calib=focal_length_calib, focal_length_corrected=focal_length_corrected, real_baseline=real_baseline)

    print(f"Hiệu chuẩn hoàn tất và các tham số đã được lưu vào {file_name}.")

# Ví dụ sử dụng hàm:
if __name__ == "__main__":
    folder_cam_left = 'calibration8/cam1'
    folder_cam_right = 'calibration8/cam2'
    calibrate_stereo_camera(folder_cam_left, folder_cam_right, pattern_size=(9, 6), square_size=2.8, real_baseline=5.5, step=25)
