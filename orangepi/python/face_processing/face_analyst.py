from .age.age_predict import AgePredictor
from .gender.gender_predict import GenderPredictor
from .landmark_face import LandmarkFace
from .emotion.emotion_predict import EmotionPredictor
from .race.race_predict import RacePredictor
import asyncio
import time
import cv2

from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

class FaceAnalyze():
    def __init__(self):
        self.face_detection = LandmarkFace()
        self.age_predictor = AgePredictor(conf=0.5)
        self.gender_predictor = GenderPredictor(conf=0.95)  
        self.emotion_predictor = EmotionPredictor()
        self.race_predictor = RacePredictor(conf=0.5)
        self.fps_avg = 0.0  # Khởi tạo self.fps
        self.call_count = 0  # Số lần gọi analyze

    async def processing_age(self, face_image):
        """Xử lý dự đoán tuổi bất đồng bộ bằng cách chạy trong luồng."""
        try:
            loop = asyncio.get_running_loop()
            age_rangeidence = await loop.run_in_executor(None, self.age_predictor.predict_age, face_image)
            return age_rangeidence
        except Exception as e:
            # Ghi lại lỗi
            logger.error(f"Lỗi trong processing_age: {e}")
            return None, None

    async def processing_gender(self, face_image):
        """Xử lý dự đoán giới tính bất đồng bộ bằng cách chạy trong luồng.""" 
        try:
            loop = asyncio.get_running_loop()
            genderidence = await loop.run_in_executor(None, self.gender_predictor.predict_gender, face_image)
            return genderidence
        except Exception as e:
            # Ghi lại lỗi
            logger.error(f"Lỗi trong processing_gender: {e}")
            return None, None
        
    async def processing_emotion(self, face_image):
        """Xử lý dự đoán tuổi bất đồng bộ bằng cách chạy trong luồng."""
        try:
            loop = asyncio.get_running_loop()
            emotionidence = await loop.run_in_executor(None, self.emotion_predictor.predict_emotion, face_image)
            return emotionidence
        except Exception as e:
            # Ghi lại lỗi
            logger.error(f"Lỗi trong processing_age: {e}")
            return None, None
        
    async def processing_race(self, face_image):
        """Xử lý dự đoán tuổi bất đồng bộ bằng cách chạy trong luồng."""
        try:
            loop = asyncio.get_running_loop()
            raceidence = await loop.run_in_executor(None, self.race_predictor.predict_race, face_image)
            return raceidence
        except Exception as e:
            # Ghi lại lỗi
            logger.error(f"Lỗi trong processing_age: {e}")
            return None, None
        
    async def analyze(self, image):
        """Phân tích khuôn mặt bất đồng bộ, đo FPS và trả về kết quả."""
        # logger.info("Starting analyze face ...")

        start_time = time.time()  # Bắt đầu đo thời gian

        # # Chạy detect_landmarks trong luồng vì nó là hàm đồng bộ
        # loop = asyncio.get_running_loop()
        # mage_rgb, landmarks, face_box = await loop.run_in_executor(None, self.face_detection.detect_landmarks, image)

        # # Kiểm tra nếu không có face_box (không phát hiện khuôn mặt)
        # if face_box is None:
        #     logger.info("Không phát hiện được khuôn mặt")
        #     return None, None

        # # Cắt vùng khuôn mặt từ ảnh
        # try:
        #     x_min, y_min, x_max, y_max = face_box
        #     face_crop = image[y_min:y_max, x_min:x_max]
        # except Exception as e:
        #     logger.error(f"Lỗi khi cắt vùng khuôn mặt: {e}")
        #     return None, None

        # Tạo các tác vụ xử lý tuổi và giới tính, chạy đồng thời
        try:
            age_task = self.processing_age(image)
            gender_task = self.processing_gender(image)
            emotion_task = self.processing_emotion(image)
            race_task = self.processing_race(image)
            age_result, gender_result, emotion_result, race_result = await asyncio.gather(age_task, gender_task, emotion_task, race_task)
        except Exception as e:
            logger.error(f"Lỗi trong face analyze: {e}")
            return None, None, None, None

        # Kết thúc đo thời gian và tính FPS
        end_time = time.time()
        duration = end_time - start_time
        fps_current = 1 / duration if duration > 0 else 0

        # Cập nhật FPS trung bình
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1
        logger.info(f"Face Analysis - Age: {age_result}, Gender: {gender_result}, Emotion: {emotion_result}, Race: {race_result}, FPS: {self.fps_avg:.2f}")

        return age_result, gender_result, emotion_result, race_result

# Ví dụ cách sử dụng (không bắt buộc) 
async def main():
    face_analyzer = FaceAnalyze()
    image_path = "4686x4000.jpg" 
    image = cv2.imread(image_path)
    while True:
        age_result, gender_result = await face_analyzer.analyze(image)
        print(f"Age: {age_result}, Gender: {gender_result}, FPS: {face_analyzer.fps_avg}")

if __name__ == "__main__":
    asyncio.run(main())