import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from utils.logging_python_orangepi import get_logger
from typing import Union, Optional

logger = get_logger(__name__)

class FeatureModel:
    def __init__(self, device: Optional[str] = 'cpu', log_level: str = 'INFO'):
        """
        Khởi tạo mô hình trích xuất đặc trưng sử dụng float32 ResNet18.

        Args:
            device (str, optional): Thiết bị chạy mô hình ('cpu' hoặc 'cuda'). Mặc định 'cpu'.
            log_level (str): Mức logging ('DEBUG', 'INFO', 'ERROR'). Mặc định 'INFO'.
        """

        # Khởi tạo thiết bị
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}") 

        # Load float32 ResNet18 với torch.jit để tối ưu inference
        try:
            base_model = models.resnet18(weights='IMAGENET1K_V1').to(self.device)
            self.feature_model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_model.eval()

            # Dùng torch.jit.trace để tối ưu inference
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device) 
            self.feature_model = torch.jit.trace(self.feature_model, dummy_input)
            logger.info("Float32 ResNet18 loaded and optimized successfully")
        except Exception as e:
            logger.error(f"Failed to load and optimize ResNet18: {str(e)}")
            raise RuntimeError(f"Failed to load ResNet18: {str(e)}")

        # Preprocessing transformation với tối ưu
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # Tối ưu resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Trích xuất đặc trưng từ ảnh đầu vào.

        Args:
            image (np.ndarray): Ảnh đầu vào dạng (H, W, C) - BGR (OpenCV).

        Returns:
            np.ndarray: Vector đặc trưng (512 chiều).

        Raises:
            RuntimeError: Nếu xử lý ảnh thất bại.
        """
        try:
            # Tối ưu chuyển đổi BGR sang RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocessing và inference
            input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_model(input_tensor)
            return features.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Failed to extract features: {str(e)}")
            raise RuntimeError(f"Failed to extract features: {str(e)}")

    def __del__(self):
        """
        Giải phóng tài nguyên khi đối tượng bị hủy.
        """
        try:
            del self.feature_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.debug("FeatureModel resources released")
        except Exception as e:
            logger.error(f"Failed to release resources: {str(e)}")