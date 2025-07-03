import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_and_save_model(model_name, save_directory):
    """
    Tải và lưu mô hình cùng bộ xử lý từ Hugging Face.

    Args:
        model_name (str): Tên mô hình trên Hugging Face.
        save_directory (str): Thư mục để lưu mô hình và các tệp cấu hình.
    """
    try:
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        os.makedirs(save_directory, exist_ok=True)
        logger.info(f"Thư mục lưu trữ: {save_directory}")

        # Tải mô hình
        logger.info(f"Đang tải mô hình {model_name}...")
        model = AutoModelForImageClassification.from_pretrained(model_name)
        logger.info("Mô hình đã được tải thành công.")

        # Tải bộ xử lý ảnh
        logger.info("Đang tải bộ xử lý ảnh...")
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        logger.info("Bộ xử lý ảnh đã được tải thành công.")

        # Lưu mô hình và bộ xử lý vào thư mục
        logger.info(f"Đang lưu mô hình và bộ xử lý vào {save_directory}...")
        model.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        logger.info("Mô hình và bộ xử lý đã được lưu thành công.")

        # Kiểm tra các tệp cần thiết
        required_files = ['config.json', 'preprocessor_config.json']
        weight_files = ['pytorch_model.bin', 'model.safetensors']
        required_files_found = True
        for file in required_files:
            file_path = os.path.join(save_directory, file)
            if os.path.exists(file_path):
                logger.info(f"Tệp {file} đã được lưu.")
            else:
                logger.error(f"Tệp {file} không được tìm thấy trong {save_directory}!")
                required_files_found = False

        # Kiểm tra tệp trọng số (ít nhất một trong hai phải tồn tại)
        weight_file_found = False
        for file in weight_files:
            file_path = os.path.join(save_directory, file)
            if os.path.exists(file_path):
                logger.info(f"Tệp trọng số {file} đã được lưu.")
                weight_file_found = True
                break

        if not weight_file_found:
            logger.error(f"Không tìm thấy tệp trọng số ({', '.join(weight_files)}) trong {save_directory}!")
            required_files_found = False

        if not required_files_found:
            raise FileNotFoundError("Một hoặc nhiều tệp cần thiết không được lưu đúng cách.")

    except Exception as e:
        logger.error(f"Lỗi khi tải hoặc lưu mô hình: {str(e)}")
        raise

if __name__ == "__main__":
    # Cấu hình
    MODEL_NAME = "prithivMLmods/open-age-detection"
    SAVE_DIRECTORY = "models"

    # Thực hiện tải và lưu mô hình
    download_and_save_model(MODEL_NAME, SAVE_DIRECTORY)