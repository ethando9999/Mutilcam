import logging
import os
from rknn.api import RKNN

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_rknn(onnx_model_path, dataset_path, rknn_output_path):
    """
    Chuyển đổi mô hình ONNX sang RKNN, sử dụng dataset để hiệu chỉnh.
    
    Args:
        onnx_model_path (str): Đường dẫn đến file ONNX.
        dataset_path (str): Đường dẫn đến file dataset.txt.
        rknn_output_path (str): Đường dẫn lưu file RKNN.
    """
    try:
        # Kiểm tra file ONNX và dataset.txt tồn tại
        if not os.path.exists(onnx_model_path):
            logger.error(f"ONNX model not found at: {onnx_model_path}")
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at: {dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Khởi tạo RKNN
        rknn = RKNN(verbose=True)
        logger.info("Initializing RKNN for RK3588")

        # Cấu hình thiết bị và chuẩn hóa dữ liệu
        logger.info("Configuring RKNN with mean and std values")
        ret = rknn.config(
            target_platform='rk3588',
            mean_values=[[127.5, 127.5, 127.5]],  # Chuẩn hóa từ [0, 255] sang [-1, 1]
            std_values=[[127.5, 127.5, 127.5]],
            quant_img_RGB2BGR=False  # Đảm bảo ảnh đầu vào không bị đảo kênh
        )
        if ret != 0:
            logger.error("Failed to configure RKNN")
            raise RuntimeError(f"RKNN config failed with code: {ret}")

        # Tải mô hình ONNX
        logger.info(f"Loading ONNX model from: {onnx_model_path}")
        ret = rknn.load_onnx(model=onnx_model_path)
        if ret != 0:
            logger.error("Failed to load ONNX model")
            raise RuntimeError(f"Load ONNX failed with code: {ret}")

        # Xây dựng mô hình với lượng hóa
        logger.info(f"Building RKNN model with dataset: {dataset_path}")
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
        if ret != 0:
            logger.error("Failed to build RKNN model")
            raise RuntimeError(f"Build RKNN failed with code: {ret}")

        # Xuất mô hình RKNN
        logger.info(f"Exporting RKNN model to: {rknn_output_path}")
        ret = rknn.export_rknn(rknn_output_path)
        if ret != 0:
            logger.error("Failed to export RKNN model")
            raise RuntimeError(f"Export RKNN failed with code: {ret}")

        # Giải phóng RKNN
        rknn.release()
        logger.info(f"RKNN model successfully saved to: {rknn_output_path}")

    except Exception as e:
        logger.error(f"Error during RKNN conversion: {str(e)}")
        raise

if __name__ == "__main__":
    # Đường dẫn tương đối từ thư mục hiện tại (~/orangepi)
    onnx_model_path = "python/models/clothes_image_detection_simplified.onnx"
    dataset_path = "python/data/calibration_images/dataset.txt"
    rknn_output_path = "rknn/clothes_image_detection.rknn"

    # Chuyển đổi mô hình
    convert_to_rknn(onnx_model_path, dataset_path, rknn_output_path)