import torch
import logging
from transformers import ViTForImageClassification

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_onnx(model_name, onnx_path):
    """
    Chuyển mô hình PyTorch sang ONNX với đầu vào cố định.
    
    Args:
        model_name (str): Tên mô hình trên Hugging Face.
        onnx_path (str): Đường dẫn lưu file ONNX.
    """
    try:
        # Tải mô hình
        logger.info(f"Loading model: {model_name}")
        model = ViTForImageClassification.from_pretrained(model_name)
        model.eval()

        # Tạo đầu vào giả
        dummy_input = torch.randn(1, 3, 224, 224)  # Batch_size = 1

        # Xuất sang ONNX
        logger.info(f"Exporting to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None  # Không dùng kích thước động
        )
        logger.info(f"ONNX model saved to: {onnx_path}")

    except Exception as e:
        logger.error(f"Error during ONNX conversion: {str(e)}")
        raise

if __name__ == "__main__":
    model_name = "dima806/fairface_age_image_detection"
    onnx_path = "orangepi/python/models/fairface_age.onnx"
    convert_to_onnx(model_name, onnx_path)