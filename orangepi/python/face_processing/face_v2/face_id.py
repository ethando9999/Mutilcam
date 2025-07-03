import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.logging_python_orangepi import get_logger
import asyncio

logger = get_logger(__name__)

# Đường dẫn tương đối đến mô hình trong thư mục models cùng cấp với face_id.py
MODEL_MAME = "mobilefacenet.pt"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)
model_path = os.path.join(MODEL_PATH, MODEL_MAME)
class MobileFaceNet:
    def __init__(self):
        """Khởi tạo MobileFaceNet với đường dẫn mô hình."""
        logger.info(f"Đang tải mô hình từ: {model_path}")
        try:
            self.model = torch.jit.load(model_path)
            self.model.eval()  # Đặt mô hình ở chế độ đánh giá để không theo dõi gradient
            logger.info("Tải mô hình thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            raise
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess(self, image):
        """Tiền xử lý ảnh trước khi embedding."""
        if image is None:
            raise ValueError("Input image is None.")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Chuyển đổi sang RGB nếu không phải
        if image.mode != 'RGB':
            logger.warning(f"Converting image from mode {image.mode} to RGB.")
            image = image.convert('RGB')
        # Kiểm tra kích thước ảnh
        if image.size[0] < 1 or image.size[1] < 1:
            raise ValueError("Input image has invalid size.")
        image = self.transform(image)
        # Kiểm tra số kênh màu
        if image.shape[0] != 3:
            raise ValueError(f"Image has {image.shape[0]} channels, expected 3.")
        image = image.unsqueeze(0)  # Thêm chiều batch
        logger.debug(f"Preprocessed image shape: {image.shape}")
        return image

    async def embed(self, image):
        """Tạo vector embedding từ ảnh khuôn mặt bất đồng bộ."""
        try:
            loop = asyncio.get_running_loop()
            processed_image = self.preprocess(image)
            # Đảm bảo batch size là 1
            if processed_image.size(0) != 1:
                raise ValueError("Batch size must be 1 for embedding.")
            with torch.no_grad():  # Ngăn chặn theo dõi gradient
                # Gọi mô hình bất đồng bộ
                embedding = await loop.run_in_executor(None, lambda: self.model(processed_image))
                # Tách tensor khỏi đồ thị tính toán và chuyển sang CPU
                embedding = embedding.detach().cpu()
                logger.debug(f"Embedding shape: {embedding.shape}, requires_grad: {embedding.requires_grad}")
            # Chuyển tensor sang NumPy
            embedding_np = embedding.squeeze().numpy()
            return embedding_np
        except Exception as e:
            logger.error(f"Lỗi trong quá trình embedding: {e}")
            return None
