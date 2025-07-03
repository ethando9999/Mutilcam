import os
import requests

# Thư mục lưu model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)
os.makedirs(MODEL_PATH, exist_ok=True)

# Danh sách file model cần tải
MODEL_FILES = [
    "shape_predictor_5_face_landmarks.dat",
    "res34_fair_align_multi_4_20190809.pt",
    "res34_fair_align_multi_7_20190809.pt",
]

BASE_URL = "https://github.com/modaccount/fairface/raw/master/models/"

def download_model_files(model_files, base_url, save_dir):
    for fname in model_files:
        url = f"{base_url}{fname}"
        dest = os.path.join(save_dir, fname)

        if os.path.exists(dest):
            print(f"✅  {fname} đã tồn tại, bỏ qua.")
            continue

        print(f"⬇️  Đang tải {fname} …")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"🎉  Đã lưu {fname} vào {dest}")
        except Exception as e:
            print(f"❌  Lỗi khi tải {fname}: {e}")

if __name__ == "__main__": 
    # Gọi hàm tải
    download_model_files(MODEL_FILES, BASE_URL, MODEL_PATH)
