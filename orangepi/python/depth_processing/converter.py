import cv2
import numpy as np
import os
import glob
import re
from tqdm import tqdm

# --- CẤU HÌNH ---
TOF_RESOLUTION = (360, 240)
DATA_TYPE = np.uint16 
MAX_DISTANCE = 4000
CONFIDENCE_THRESHOLD = 30 

# THƯ MỤC DỮ LIỆU NHẬN ĐƯỢC TRÊN OPI
INPUT_DEPTH_DIR = "depth_frames"
INPUT_AMP_DIR = "amplitude_frames"
INPUT_CONF_DIR = "confidence_frames"

# THƯ MỤC DỮ LIỆU GỐC (SAO CHÉP TỪ PI SANG)
GROUND_TRUTH_DIR = "pi02_raw_data" 

# THƯ MỤC ĐẦU RA
OUTPUT_VISUAL_DIR = "opi_visual_data"
IMAGE_FORMAT = ".png"

def read_bin_frame(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return np.frombuffer(data, dtype=DATA_TYPE).reshape((TOF_RESOLUTION[1], TOF_RESOLUTION[0]))
    except Exception as e:
        print(f"Error reading .bin file {file_path}: {e}")
        return None

# --- SỬA LỖI: Thêm các hàm visualize bị thiếu ---
def visualize_depth_pi(depth_frame, confidence_frame):
    """Chuyển ảnh depth thành ảnh màu giả, lọc nhiễu, dùng colormap RAINBOW."""
    depth_8bit = np.clip(depth_frame, 0, MAX_DISTANCE)
    depth_8bit = (depth_8bit / MAX_DISTANCE * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_RAINBOW)
    if confidence_frame is not None:
        colored_depth[confidence_frame < CONFIDENCE_THRESHOLD] = (0, 0, 0)
    return colored_depth

def visualize_amplitude_pi(amplitude_frame, confidence_frame):
    """Chuyển ảnh amplitude thành ảnh xám, lọc nhiễu."""
    amplitude_frame = np.clip(amplitude_frame, 0, 4095) 
    amp_8bit = (amplitude_frame / 4095.0 * 255).astype(np.uint8)
    if confidence_frame is not None:
        amp_8bit[confidence_frame < CONFIDENCE_THRESHOLD] = 0
    return amp_8bit
# --- KẾT THÚC SỬA LỖI ---

def main():
    print("--- Starting Verification & Conversion Process v3.1 (Fixed) ---")

    if not os.path.isdir(GROUND_TRUTH_DIR):
        print(f"\nFATAL ERROR: Ground truth directory '{GROUND_TRUTH_DIR}' not found!")
        print("Please copy the 'pi02_raw_data' folder from the Raspberry Pi to this directory first.")
        return

    os.makedirs(OUTPUT_VISUAL_DIR, exist_ok=True)

    get_id = lambda f: int(re.search(r'frame_(\d+).', os.path.basename(f)).group(1))
    
    received_map = {get_id(f): f for f in glob.glob(os.path.join(INPUT_DEPTH_DIR, '*.bin'))}
    ground_truth_map = {get_id(f): f for f in glob.glob(os.path.join(GROUND_TRUTH_DIR, 'depth_frame_*.npy'))}

    common_ids = sorted(list(set(received_map.keys()) & set(ground_truth_map.keys())))
    
    if not common_ids:
        print("\nError: No common frames found between received data and ground truth.")
        return

    print(f"\nFound {len(common_ids)} frames to verify and process...")
    verification_passed = True

    for frame_id in tqdm(common_ids, desc="Verifying and Converting"):
        # Đọc dữ liệu đã nhận (.bin)
        received_depth = read_bin_frame(os.path.join(INPUT_DEPTH_DIR, f"frame_{frame_id}.bin"))
        
        # Đọc dữ liệu gốc (.npy)
        gt_depth = np.load(os.path.join(GROUND_TRUTH_DIR, f"depth_frame_{frame_id}.npy"))

        # --- BƯỚC QUAN TRỌNG NHẤT: ĐỐI CHIẾU ---
        if np.array_equal(received_depth, gt_depth):
            pass
        else:
            print(f"\n\n!!! VERIFICATION FAILED FOR FRAME {frame_id} !!!")
            print("Data received over UDP does not match the ground truth saved on Pi.")
            verification_passed = False
            break 
        
        # Nếu đối chiếu thành công, tiếp tục xử lý
        received_amp = read_bin_frame(os.path.join(INPUT_AMP_DIR, f"frame_{frame_id}.bin"))
        received_conf = read_bin_frame(os.path.join(INPUT_CONF_DIR, f"frame_{frame_id}.bin"))
        
        if received_amp is None or received_conf is None: continue

        # Tạo ảnh trực quan từ dữ liệu đã nhận
        # Giờ đây các hàm đã được định nghĩa ở trên và có thể gọi được
        vis_depth = visualize_depth_pi(received_depth, received_conf)
        cv2.imwrite(os.path.join(OUTPUT_VISUAL_DIR, f"depth_image_{frame_id}.png"), vis_depth)
        
        vis_amp = visualize_amplitude_pi(received_amp, received_conf)
        cv2.imwrite(os.path.join(OUTPUT_VISUAL_DIR, f"amp_image_{frame_id}.png"), vis_amp)
    
    print("\n--- Verification and Conversion Complete ---")
    if verification_passed:
        print("✅ SUCCESS: All received frames match the ground truth data perfectly.")
        print("The problem is NOT in the UDP sending/receiving process.")
        print(f"Visual results are saved in '{OUTPUT_VISUAL_DIR}'")
    else:
        print("❌ FAILURE: Data corruption occurred during UDP transmission.")

if __name__ == "__main__":
    # SỬA LỖI: Xóa các dòng thừa, chỉ gọi hàm main()
    main() 
