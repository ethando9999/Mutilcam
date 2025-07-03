import logging
import onnx
import onnxsim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đơn giản hóa mô hình ONNX
logger.info("Simplifying ONNX model")
try:
    model_path = "orangepi/python/models/fairface_age.onnx"
    simplified_model, check = onnxsim.simplify(model_path)
    if check:
        onnx.save(simplified_model, "orangepi/python/models/clothes_image_detection_simplified.onnx")
        logger.info("Simplified model saved to clothes_image_detection_simplified.onnx")
    else:
        logger.error("Failed to simplify ONNX model")
        raise ValueError("ONNX simplification failed")
except Exception as e:
    logger.error(f"Failed to simplify ONNX: {str(e)}")
    raise