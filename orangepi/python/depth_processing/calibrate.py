# calibrate.py (version 5.4 - Unified & Robust)
import numpy as np
import cv2
import os
import glob
import re # Sử dụng Regular Expressions cho sự linh hoạt
from logging_config import setup_logging, get_logger

# --- CẤU HÌNH LOGGING ---
setup_logging()
logger = get_logger(__name__)

# --- CẤU HÌNH HIỆU CHỈNH ---
CHESSBOARD_DIMS = (9, 6)
SQUARE_SIZE_MM = 40.0

# --- CẤU HÌNH XỬ LÝ ---
VISUALIZE = True # Bật/tắt cửa sổ xem trước, rất hữu ích để gỡ lỗi

# --- CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RGB_DIR = os.path.join(SCRIPT_DIR, "rgb_frames")
AMP_DIR = os.path.join(SCRIPT_DIR, "amplitude_frames")
TOF_INTRINSICS_FILE = os.path.join(SCRIPT_DIR, "tof_intrinsics.npz")
STEREO_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "stereo_calib_result.npz")

# [TỪ v5.3] Định nghĩa mẫu (pattern) cho tên file RGB bằng Regex
# Giúp xử lý các tên file như 'rgb_001.png', 'rgb_001_x.png', 'rgb_001_whatever.png'
RGB_FILENAME_PATTERN = re.compile(r"rgb_(\d+)(.*)\.png")

def check_gui_support():
    try: 
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        return True
    except cv2.error:
        return False  

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Tính toán sai số chiếu lại trung bình trên từng ảnh."""
    per_view_errors = []
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        per_view_errors.append(error)
    return per_view_errors

def main():
    logger.info("="*50)
    logger.info("Starting Stereo Calibration Process (v5.4 - Unified & Robust)")
    logger.info("="*50)

    # 1. TẢI THÔNG SỐ NỘI TẠI CỦA CAMERA TOF
    logger.info(f"Attempting to load ToF intrinsics from '{TOF_INTRINSICS_FILE}'...")
    if not os.path.exists(TOF_INTRINSICS_FILE):
        logger.error(f"ToF intrinsics file not found! Please run 'get_tof_intrinsics.py' first.")
        return
    with np.load(TOF_INTRINSICS_FILE) as data:
        mtx_tof, dist_tof = data['mtx_tof'], data['dist_tof']
    logger.info("Successfully loaded ToF camera factory intrinsics.")

    # 2. CHUẨN BỊ ĐIỂM VÀ TÌM GÓC BÀN CỜ
    objp = np.zeros((CHESSBOARD_DIMS[0] * CHESSBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_DIMS[0], 0:CHESSBOARD_DIMS[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM

    objpoints, imgpoints_rgb, imgpoints_amp, image_paths_used = [], [], [], []
    rgb_shape, amp_shape = None, None

    rgb_images = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    logger.info(f"Found {len(rgb_images)} RGB images. Processing all.")
    
    gui_enabled = VISUALIZE and check_gui_support()
    if VISUALIZE and not gui_enabled:
        logger.warning("GUI is not supported in this environment. Visualization will be disabled.")

    for rgb_path in rgb_images:
        rgb_filename = os.path.basename(rgb_path)
        
        # [TỪ v5.3] Sử dụng Regex để xử lý tên file linh hoạt
        match = RGB_FILENAME_PATTERN.match(rgb_filename)
        if not match:
            logger.warning(f"Skipping file with unexpected name format: {rgb_filename}")
            continue

        numeric_id = match.group(1) # -> '001'
        suffix = match.group(2)     # -> '_x' hoặc ''
        full_id_str = f"{numeric_id}{suffix}" # -> '001_x' hoặc '001'
        
        amp_filename = f"amp_{numeric_id}{suffix}.png"
        amp_path = os.path.join(AMP_DIR, amp_filename)

        if not os.path.exists(amp_path):
            logger.warning(f"Skipping ID {full_id_str}: Corresponding amplitude image '{amp_filename}' not found.")
            continue
            
        img_rgb = cv2.imread(rgb_path)
        img_amp = cv2.imread(amp_path, cv2.IMREAD_GRAYSCALE)
        if rgb_shape is None: rgb_shape = img_rgb.shape[:2][::-1]
        if amp_shape is None: amp_shape = img_amp.shape[:2][::-1]
        
        ret_rgb, corners_rgb = cv2.findChessboardCorners(img_rgb, CHESSBOARD_DIMS, None)
        ret_amp, corners_amp = cv2.findChessboardCorners(img_amp, CHESSBOARD_DIMS, None)

        if gui_enabled:
            img_rgb_display = cv2.drawChessboardCorners(img_rgb.copy(), CHESSBOARD_DIMS, corners_rgb, ret_rgb)
            status_str = f"RGB: {'OK' if ret_rgb else 'FAIL'}, ToF: {'OK' if ret_amp else 'FAIL'}"
            cv2.imshow(f'ID: {full_id_str} - Status: {status_str}', img_rgb_display)
            cv2.waitKey(200)

        # [TỪ v5.2] Logic kiểm tra và ghi log chi tiết
        if ret_rgb and ret_amp:
            logger.info(f"  -> Success: Found chessboard on both images for ID: {full_id_str}")
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            imgpoints_rgb.append(cv2.cornerSubPix(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), corners_rgb, (11,11), (-1,-1), criteria))
            imgpoints_amp.append(cv2.cornerSubPix(img_amp, corners_amp, (11,11), (-1,-1), criteria))
            image_paths_used.append(rgb_filename)
        else:
            error_details = []
            if not ret_rgb:
                error_details.append("RGB image")
            if not ret_amp:
                error_details.append("ToF amplitude image")
            logger.warning(f"  -> Fail: Could not find chessboard on {' and '.join(error_details)} for ID: {full_id_str}")

    if gui_enabled: cv2.destroyAllWindows()
    
    if len(objpoints) < 10:
        logger.error(f"Insufficient valid image pairs ({len(objpoints)} found). Need at least 10 for a reliable calibration. Aborting.")
        return

    # 3. HIỆU CHỈNH NỘI TẠI CHO CAMERA RGB
    logger.info(f"Using {len(objpoints)} valid image pairs for calibration.")
    logger.info("Calibrating RGB camera intrinsics...")
    ret_rgb, mtx_rgb, dist_rgb, rvecs_rgb, tvecs_rgb = cv2.calibrateCamera(objpoints, imgpoints_rgb, rgb_shape, None, None)

    # 4. HIỆU CHỈNH STEREO
    logger.info("Performing Stereo Calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret, mtx_rgb, dist_rgb, mtx_tof, dist_tof, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_rgb, imgpoints_amp, mtx_rgb, dist_rgb, mtx_tof, dist_tof,
        rgb_shape, criteria=criteria, flags=flags
    )

    # 5. PHÂN TÍCH VÀ LƯU KẾT QUẢ (Log chi tiết như bạn mong muốn)
    logger.info("="*50)
    logger.info("--- STEREO CALIBRATION RESULTS ---")
    logger.info("="*50)
    logger.info(f"Overall Reprojection Error: {ret:.4f}")
    
    logger.info("\n--- Detailed Per-Image Reprojection Error Analysis ---")
    errors_rgb = calculate_reprojection_error(objpoints, imgpoints_rgb, rvecs_rgb, tvecs_rgb, mtx_rgb, dist_rgb)
    errors_tof = []
    for i in range(len(objpoints)):
        _, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints_amp[i], mtx_tof, dist_tof)
        imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx_tof, dist_tof)
        error = cv2.norm(imgpoints_amp[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        errors_tof.append(error)
    logger.info("Errors per image pair (RGB | ToF):")
    for i in range(len(errors_rgb)):
        logger.info(f"  - Pair {image_paths_used[i]:<20}: {errors_rgb[i]:.4f} | {errors_tof[i]:.4f}")

    logger.info("\n--- Matrix Parameters ---")
    logger.info(f"RGB Camera Matrix (K1):\n{mtx_rgb}")
    logger.info(f"ToF Camera Matrix (K2):\n{mtx_tof}")
    logger.info(f"Rotation Matrix (R):\n{R}")
    logger.info(f"Translation Vector (T) in mm:\n{T}")

    np.savez(STEREO_OUTPUT_FILE, 
             mtx_rgb=mtx_rgb, dist_rgb=dist_rgb, mtx_tof=mtx_tof, dist_tof=dist_tof,
             R=R, T=T, error=ret)
    logger.info(f"\n✅ Calibration results saved to '{os.path.abspath(STEREO_OUTPUT_FILE)}'")

if __name__ == '__main__':
    main()