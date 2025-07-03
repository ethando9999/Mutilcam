import subprocess
subprocess.run("export PYTHONPATH=/home/rpi5/python_rpi/python/feature:$PYTHONPATH", shell=True)

import torch
import cv2
import numpy as np
import os
from PIL import Image
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

current_path = os.path.dirname(os.path.abspath(__file__))

# Nếu model của bạn là Duke thì DATASET nên đặt tương ứng, ví dụ "DukeMTMC" hoặc folder config tương ứng.
MODEL_NAME = "market_sbs_R101-ibn.pth"
CONFIG_NAME = "sbs_R101-ibn.yml"
DATASET = "Market1501"  # đảm bảo folder fastreid/configs/DukeMTMC/sbs_S50.yml tồn tại
MODEL_PATH = os.path.join(current_path, "models", MODEL_NAME)
CONFIG_PATH = os.path.join(current_path, "fastreid", "configs", DATASET, CONFIG_NAME)  

class FeatureModel:
    def __init__(self, config_file=CONFIG_PATH, model_weights=MODEL_PATH, device="cpu"):
        try:
            # 1. Load config và weights
            self.cfg = get_cfg()
            self.cfg.merge_from_file(config_file) 
            self.cfg.MODEL.WEIGHTS = model_weights 
            # Chuyển device (cpu hoặc "cuda")
            self.cfg.MODEL.DEVICE = device 
            self.cfg.freeze()

            # 2. Khởi tạo predictor
            self.predictor = DefaultPredictor(self.cfg)
            # DefaultPredictor bên FastReID sẽ set model.eval() và handle device.

            # 3. Build transforms chuẩn inference
            # build_transforms trả về một Callable: PIL Image -> Tensor đã normalize theo cfg
            self.transforms = build_transforms(self.cfg, is_train=False)

            logger.info("[FeatureModel] Initialized successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize FeatureModel: {e}")
            raise

    def extract_feature(self, img):
        """
        Trích xuất đặc trưng từ ảnh đầu vào (numpy array BGR hoặc PIL.Image).
        Trả về: numpy.ndarray shape (1, feature_dim) sau L2-normalize.
        """
        try:
            # 1. Chuyển PIL -> numpy BGR, hoặc kiểm tra numpy
            if isinstance(img, Image.Image):
                # PIL cung cấp RGB, nhưng model FastReID thường xử lý BGR.
                img = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR

            if not isinstance(img, np.ndarray):
                raise ValueError(f"Invalid image type: {type(img)}. Cần numpy.ndarray hoặc PIL.Image.")

            # 2. Đảm bảo img là BGR uint8
            if img.dtype != np.uint8:
                # Chuyển sang uint8 nếu cần: giả sử img đã ở [0,255]
                img = img.astype(np.uint8)

            # 3. Resize theo cfg(INPUT.SIZE_TEST)
            # cfg.INPUT.SIZE_TEST thường là [H, W]
            size_test = self.cfg.INPUT.SIZE_TEST
            if isinstance(size_test, (list, tuple)) and len(size_test) == 2:
                h, w = size_test
                # OpenCV resize dùng (width, height)
                img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # Nếu SIZE_TEST không phải [H, W], bạn có thể bỏ qua resize hoặc tùy chỉnh.
                img_resized = img

            # 4. Chuyển BGR -> RGB rồi thành PIL để build_transforms nếu cần
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # 5. Áp dụng transforms của FastReID (ToTensor + Normalize dựa trên cfg)
            tensor = self.transforms(img_pil)  # kết quả: Tensor [3, H, W], float32
            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise RuntimeError(f"Unexpected tensor shape after transforms: {tensor.shape}")

            # 6. Đưa vào model
            input_tensor = tensor.unsqueeze(0).to(self.cfg.MODEL.DEVICE)  # [1, 3, H, W]
            with torch.no_grad():
                feat = self.predictor.model(input_tensor)  # output shape: [1, D]
                # 7. L2-normalize embedding
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)

            return feat.cpu().numpy()  # shape (1, D)
        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed: {e}")
            return None

# Ví dụ sử dụng:
# fm = FeatureModel(device="cuda" if torch.cuda.is_available() else "cpu")
# img = cv2.imread("/path/to/image.jpg")  # BGR uint8
# feat = fm.extract_feature(img)
# print(feat.shape)
