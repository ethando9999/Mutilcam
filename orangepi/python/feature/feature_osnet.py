import os
from pathlib import Path
import torch
from torchreid import models
from torchreid.reid.utils import load_pretrained_weights
from torchreid.reid_model_factory import get_model_url
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

MODEL_NAME = "osnet_ain_x1_0"
MODEL_FILE = "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
NUM_CLASSES = 4101
IMG_HEIGHT = 256
IMG_WIDTH = 128


class FeatureModel:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cpu")

        # Xác định đường dẫn đến checkpoint
        current_path = Path(__file__).parent
        model_dir = current_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        if model_path is None:
            model_path = model_dir / MODEL_FILE
        else:
            model_path = Path(model_path)

        # Build model architecture
        model = models.build_model(
            name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            loss='softmax'
        )

        # Nếu checkpoint không tồn tại, thì tải từ get_model_url
        if not model_path.is_file():
            logger.warning(f"Checkpoint {model_path.name} không tồn tại. Đang tải tự động...")
            try:
                tmp_model = models.build_model(name=MODEL_NAME, num_classes=NUM_CLASSES, loss='softmax')
                url = get_model_url(tmp_model)
                if url:
                    torch.hub.download_url_to_file(url, str(model_path))
                    logger.info(f"Tải thành công: {model_path}")
                else:
                    logger.error("Không tìm thấy URL cho pretrained model.")
            except Exception as e:
                logger.error(f"Lỗi khi tải model: {e}")

        # Load checkpoint nếu có
        if model_path.is_file():
            try:
                load_pretrained_weights(model, str(model_path))
                logger.info(f"Đã load weights từ {model_path}")
            except Exception as e:
                logger.error(f"Lỗi khi load weights từ {model_path}: {e}")
        else:
            logger.error("Không tìm thấy hoặc không tải được model. Dùng model ngẫu nhiên.")

        model.to(self.device)
        model.eval()
        self.model = model

        self.preprocess = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got {type(image)}")

            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape {image.shape}, expected (H, W, 3)")

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            inp = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(inp)

            feat = feat.squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat /= norm

            return feat

        except Exception as e:
            logger.error(f"Error in extract_feature: {e}")
            return None
