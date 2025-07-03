from utils.logging_python_orangepi import get_logger
logger = get_logger(__name__)

from .mdp_face_detection import FaceDetection
from .dlib_aligner import FaceAligner
from .fairface import FairFacePredictor
from .gender_predictor import GenderClassifier
from .age_predictor import AgePredictor
from .face_id import MobileFaceNet
import asyncio
import time
import cv2
import os
import glob

from config import FACE_CONFIG

class FaceAnalyze:
    def __init__(self):
        conf = FACE_CONFIG
        self.face_conf = conf.get("face_detection_conf", 0.8)
        self.race_conf = conf.get("race_conf", 0.5)
        self.gender_conf = conf.get("gender_conf", 0.8)
        self.age_conf = conf.get("age_conf", 0.5)

        self.face_detection = FaceDetection(min_detection_confidence=self.face_conf)
        self.fairface_predictor = FairFacePredictor(
            model_selection=0,
            r_conf=self.race_conf,
            g_conf=self.gender_conf,
            a_conf=self.age_conf
        )
        self.face_aligner = FaceAligner()
        self.face_id = MobileFaceNet()
        self.fps_avg = 0.0
        self.call_count = 0


    async def processing_face(self, face_image):  
        """
        Asynchronous wrapper for FairFacePredictor.predict.
        Returns the embedding and prediction dict or None on error.
        """
        try:
            loop = asyncio.get_running_loop()
            embedding_task = self.face_id.embed(face_image)
            fairface_task = loop.run_in_executor(None, self.fairface_predictor.predict, face_image)
            embedding, fairface_result = await asyncio.gather(embedding_task, fairface_task)
            return embedding, fairface_result
        except Exception as e:
            logger.error(f"Error in processing_face (FairFace): {e}")
            return None, None

    def detect_and_align(self, image, margin: float = 0.4, padding: float = 0.25): 
        """
        Detect and align faces in the image.

        Args:
            image (numpy.ndarray): Input image in BGR format.
            margin (float): Margin around the detected face bounding box.
            padding (float): Padding for face alignment.

        Returns:
            aligned_face (numpy.ndarray): Aligned face image, or None if no face is detected.
        """
        infos, _ = self.face_detection.detect(image)
        if not infos:
            return None

        bbox = infos[0]['bbox']
        h, w = image.shape[:2]
        x1 = max(0, int((bbox.xmin * w) - bbox.width * w * margin))
        y1 = max(0, int((bbox.ymin * h) - bbox.height * h * margin))
        x2 = min(w, int(x1 + bbox.width * w * (1 + 2 * margin)))
        y2 = min(h, int(y1 + bbox.height * h * (1 + 2 * margin)))

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        return self.face_aligner.aligning(roi, padding=padding)

    async def analyze(self, image):
        """
        Analyze the face in the image, including detection, alignment, and attribute prediction.

        Args:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            tuple: (embedding, age, gender, None, race) 
                or (None, None, None, None, None) if no face is detected or an error occurs.
        """
        start_time = time.time()

        # Bước 1: Phát hiện và căn chỉnh khuôn mặt
        face_chip = self.detect_and_align(image)
        if face_chip is None:
            logger.info("No face detected in analyze")
            return None, None, None, None, None

        # Bước 2: Xử lý nhận diện và phân tích khuôn mặt
        try:
            face_embedding, face_result = await self.processing_face(face_chip)
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return None, None, None, None, None

        # Nếu không có embedding thì không tiếp tục
        if face_embedding is None:
            return None, None, None, None, None

        # Dù face_result có thể None, vẫn tiếp tục trả embedding
        age = face_result.get("age") if face_result else None
        gender = face_result.get("gender") if face_result else None
        race = face_result.get("race") if face_result else None

        # Bước 3: Tính toán FPS trung bình
        duration = time.time() - start_time
        fps_current = 1 / duration if duration > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1

        logger.info(f"Face Analysis - FairFace: {face_result}, FPS: {self.fps_avg:.2f}")

        return face_embedding, age, gender, None, race


def natural_keys(text):
    """
    Hàm sắp xếp theo thứ tự tự nhiên (natural sort).
    """
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

def main():
    face_analyze = FaceAnalyze()
    folder = "python/face_processing/"
    images_list = sorted(glob.glob(os.path.join(folder, '*.jpg')), key=natural_keys)
    try:
        for imagepath in images_list:
            image = cv2.imread(imagepath)
            if image is None:
                logger.error(f"Không thể đọc ảnh từ {imagepath}")
                continue
            
            # Chạy phân tích bất đồng bộ
            loop = asyncio.get_event_loop()
            age, gender, _, race = loop.run_until_complete(face_analyze.analyze(image))
            
            # In kết quả ra console
            if age is not None: 
                print(f"Age: {age}, Gender: {gender}, Race: {race}, FPS: {face_analyze.fps_avg:.2f}")
            else:
                print(f"Không phát hiện được khuôn mặt hoặc lỗi trong quá trình xử lý: {imagepath}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        
if __name__ == "__main__": 
    main()