import asyncio
import time
import cv2
import os

from utils.logging_python_orangepi import get_logger
from .mdp_face_detection import FaceDetection
from .dlib_aligner import FaceAligner
from .fairface import FairFacePredictor
from .gender_vit_fairface import GenderClassifier
from .age_predictor import AgePredictor
from .face_id import MobileFaceNet
from config import FACE_CONFIG

logger = get_logger(__name__)

class FaceAnalyze:
    def __init__(self):
        conf = FACE_CONFIG
        # Thresholds
        self.face_conf   = conf.get("face_detection_conf", 0.8)
        self.race_conf   = conf.get("race_conf", 0.5)
        self.gender_conf = conf.get("gender_conf", 0.8)
        self.age_conf    = conf.get("age_conf", 0.5)

        # Core modules
        self.face_detection = FaceDetection(min_detection_confidence=self.face_conf)
        self.face_aligner    = FaceAligner()
        self.face_id         = MobileFaceNet()

        # Predictors
        self.race_predictor   = FairFacePredictor(model_selection=1, r_conf=self.race_conf)
        self.gender_predictor = GenderClassifier(threshold=self.gender_conf)
        self.age_predictor    = AgePredictor(threshold=self.age_conf)

        # FPS tracking
        self.fps_avg    = 0.0
        self.call_count = 0

    async def processing_face(self, face_chip, face_roi):
        """
        Asynchronously compute face embedding and attribute predictions in parallel.
        Returns:
            embedding, age_label, gender_label, emotion, race_label
        """
        loop = asyncio.get_running_loop()
        # Launch tasks concurrently
        embed_task  = self.face_id.embed(face_chip)
        race_task   = loop.run_in_executor(None, self.race_predictor.predict, face_chip)
        gender_task = loop.run_in_executor(None, self.gender_predictor.predict_top, face_roi)
        age_task    = loop.run_in_executor(None, self.age_predictor.predict_top, face_roi)

        embedding, race_res, gender_res, age_res = await asyncio.gather(
            embed_task, race_task, gender_task, age_task
        )

        # Unpack predictions
        race_label, _   = race_res.get('race', (None, 0.0))
        gender_label, _ = gender_res
        age_label, _    = age_res

        return embedding, age_label, gender_label, None, race_label

    def detect_and_align(self, image, margin: float = 0.4, padding: float = 0.25):
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

        return self.face_aligner.aligning(roi, padding=padding), roi

    async def analyze(self, image):
        """
        Complete async face analysis pipeline.
        Returns:
            embedding, age_label, gender_label, emotion, race_label
            or (None, None, None, None, None) on failure.
        """
        start = time.time()

        # Detect & align
        face_chip, face_roi = self.detect_and_align(image)
        if face_chip is None:
            logger.info("No face detected in analyze")
            return None, None, None, None, None

        # Process face attributes asynchronously
        try:
            embedding, age_label, gender_label, emotion, race_label = await self.processing_face(face_chip, face_roi)
        except Exception as e:
            logger.error(f"Error in processing_face: {e}")
            return None, None, None, None, None

        if embedding is None:
            return None, None, None, None, None

        # FPS calculation
        duration    = time.time() - start
        fps_current = 1.0 / duration if duration > 0 else 0.0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1

        # Logging
        logger.info(
            f"FaceAnalysis | Race: {race_label}, Gender: {gender_label}, "
            f"Age: {age_label}, Emotion: {emotion}, FPS: {self.fps_avg:.2f}"
        )

        return embedding, age_label, gender_label, emotion, race_label


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