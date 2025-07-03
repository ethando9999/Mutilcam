import logging
import os

current_directory = os.getcwd()
# Cấu hình logging
LOG_FILE = os.path.join(current_directory, 'pi_client/cam.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Ghi log vào file
        logging.StreamHandler()        # Hiển thị log trong terminal
    ]
)
