import torch
import cv2
import logging
from id.feature_extraction import FeatureModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        model = FeatureModel(device='cpu')
        image = cv2.imread('python/face_proccesing/face.jpg')
        if image is None:
            raise RuntimeError("Failed to load test image")
        import time
        start = time.time()
        features = model.extract_features(image)
        fps = 1 / (time.time() - start)
        logger.info(f"Features shape: {features.shape}, FPS: {fps:.2f}")
    except Exception as e:
        logger.error(f"Error: {e}")