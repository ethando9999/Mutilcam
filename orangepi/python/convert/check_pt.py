import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải mô hình và processor
logger.info("Loading model and processor from Hugging Face")
model_name = "dima806/clothes_image_detection"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
model.eval()

# Kiểm tra cấu hình mô hình
logger.info(f"Model config: {model.config}")

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]  # Shape: [1, 3, 224, 224]

# Kiểm tra inference
image_path = "orangepi/object_detection/image.webp"  # Thay bằng đường dẫn ảnh của bạn
pixel_values = preprocess_image(image_path)
with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    class_labels = model.config.id2label
    logger.info(f"Predicted class: {class_labels[predicted_class]}")