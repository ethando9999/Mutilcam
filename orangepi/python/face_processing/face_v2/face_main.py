import os
import cv2
import glob
import asyncio
import time
from datetime import datetime

from .mdp_face_detection import FaceDetection
from .dlib_aligner import FaceAligner
from .fairface import FairFacePredictor
from .face_id import MobileFaceNet

from utils.logging_python_orangepi import get_logger
from config import FACE_CONFIG, OPI_CONFIG

logger = get_logger(__name__)


def natural_keys(text):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]


class FaceAnalyze:
    def __init__(self):
        conf = FACE_CONFIG
        self.face_conf = conf.get("face_detection_conf", 0.8)
        self.race_conf = conf.get("race_conf", 0.5)
        self.gender_conf = conf.get("gender_conf", 0.8)
        self.age_conf = conf.get("age_conf", 0.5)

        self.face_detector = FaceDetection(min_detection_confidence=self.face_conf)
        self.face_aligner = FaceAligner()
        self.fairface = FairFacePredictor(
            model_selection=0,
            r_conf=self.race_conf,
            g_conf=self.gender_conf,
            a_conf=self.age_conf
        )
        self.face_id = MobileFaceNet()

        self.fps_avg = 0.0
        self.call_count = 0
        logger.info("Init FaceAnalyze successfully!")

    async def detect_and_align(self, image, margin: float = 0.4, padding: float = 0.25):
        """Run CPU-bound detection+alignment in thread."""
        def _sync_detect_align(img):
            infos, _ = self.face_detector.detect(img)
            if not infos:
                return None
            bbox = infos[0]['bbox']
            h, w = img.shape[:2]
            x1 = max(0, int((bbox.xmin * w) - bbox.width * w * margin))
            y1 = max(0, int((bbox.ymin * h) - bbox.height * h * margin))
            x2 = min(w, int(x1 + bbox.width * w * (1 + 2 * margin)))
            y2 = min(h, int(y1 + bbox.height * h * (1 + 2 * margin)))
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            return self.face_aligner.aligning(roi, padding=padding)

        return await asyncio.to_thread(_sync_detect_align, image)

    async def save_face_chip(self, face_chip, base_name: str = None):
        """Save face chip asynchronously."""
        if face_chip is None:
            return
        def _save():
            save_dir = os.path.join(OPI_CONFIG.get("results_dir"), "face_chip")
            os.makedirs(save_dir, exist_ok=True)
            name = base_name or f"face_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            path = os.path.join(save_dir, name)
            cv2.imwrite(path, face_chip)
            logger.info(f"Saved face_chip to: {path}")

        await asyncio.to_thread(_save)

    async def processing_face(self, face_chip):
        """Run embedding and FairFace predict in parallel."""
        loop = asyncio.get_running_loop()
        embed_task = self.face_id.embed(face_chip)
        fair_task = loop.run_in_executor(None, self.fairface.predict, face_chip)
        embedding, fairface_result = await asyncio.gather(embed_task, fair_task)
        return embedding, fairface_result

    async def analyze(self, image):
        """Full pipeline: detect, align, save chip, predict attributes."""
        start = time.time()

        face_chip = await self.detect_and_align(image)
        if face_chip is None:
            logger.info("No face detected")
            return None, None, None, None, None

        # asynchronously save chip but don't await to speed up
        asyncio.create_task(self.save_face_chip(face_chip))

        try:
            embedding, result = await self.processing_face(face_chip)
        except Exception as e:
            logger.error(f"Error in processing_face: {e}")
            return None, None, None, None, None

        if embedding is None:
            return None, None, None, None, None

        age = result.get("age")
        gender = result.get("gender")
        race = result.get("race")

        # update FPS
        dur = time.time() - start
        fps = 1 / dur if dur > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps) / (self.call_count + 1)
        self.call_count += 1
        logger.info(f"Face Analyze: age={age}, gender={gender}, race={race}, fps={self.fps_avg:.2f}")

        return embedding, age, gender, None, race


def main():
    face_analyzer = FaceAnalyze()
    folder = "python/face_processing/"
    images = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=natural_keys)

    async def _run():
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb, age, gender, _, race = await face_analyzer.analyze(img)
            print(f"{img_path}: age={age}, gender={gender}, race={race}, fps={face_analyzer.fps_avg:.2f}")

    asyncio.run(_run())

if __name__ == "__main__":
    main()
