import dlib
print("Đường dẫn module dlib:", dlib.__file__)
import cv2
import os
import numpy as np
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)


# Định nghĩa tên file mô hình
SHAPE = "shape_predictor_5_face_landmarks.dat"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)
# print(MODEL_PATH)
class FaceAligner:
    def __init__(self, model_dir=MODEL_PATH):
        """
        Khởi tạo FaceAligner với mô hình shape predictor của dlib.

        Args:
            model_dir (str): Đường dẫn đến thư mục chứa mô hình.
        """
        self.model_dir = model_dir
        try:
            # Tải shape predictor
            self.sp = dlib.shape_predictor(os.path.join(model_dir, SHAPE))
        except Exception as e:
            logger.error(f"Không thể tải mô hình dlib từ {model_dir}: {e}")
            raise

    def detect_facechip(self, image_np, size=300, padding=0.25):
        """
        Phát hiện và căn chỉnh khuôn mặt từ ảnh đầu vào sử dụng Mediapipe Face Detection.

        Args:
            image_np (np.ndarray): Ảnh đầu vào (BGR).
            size (int): Kích thước đầu ra của face chip.
            padding (float): Độ mở rộng xung quanh khuôn mặt.

        Returns:
            np.ndarray: Face chip đã căn chỉnh (RGB) hoặc None nếu không phát hiện được.
        """
        # Nhập FaceDetection trong hàm để tránh vòng lặp nhập
        from mdp_aligner import FaceDetection
        # Khởi tạo detector từ Mediapipe
        detector = FaceDetection()
        # Phát hiện khuôn mặt
        infos, _ = detector.detect(image_np)
        if not infos:
            return None

        # Lấy bbox đầu tiên
        bbox = infos[0]['bbox']
        h, w = image_np.shape[:2]
        x1 = max(0, int((bbox.xmin * w) - bbox.width * w * 0.25))
        y1 = max(0, int((bbox.ymin * h) - bbox.height * h * 0.25))
        x2 = min(w, int(x1 + bbox.width * w * (1 + 0.5)))
        y2 = min(h, int(y1 + bbox.height * h * (1 + 0.5)))

        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Căn chỉnh ROI bằng dlib
        return self.aligning(roi, size=size, padding=padding)

    def aligning(self, face: np.ndarray, size=300, padding=0.25):
        """
        Căn chỉnh một vùng khuôn mặt đã được cắt sẵn (ROI).

        Args:
            face (np.ndarray): Mảng numpy của vùng khuôn mặt (BGR hoặc RGB).
            size (int): Kích thước đầu ra của face chip.
            padding (float): Độ mở rộng xung quanh khuôn mặt.

        Returns:
            np.ndarray: Face chip đã căn chỉnh (RGB) hoặc None nếu thất bại.
        """
        # Đảm bảo ảnh ở định dạng RGB cho dlib
        if face.shape[2] == 3 and face.dtype == np.uint8:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face
        
        # Tạo rectangle bao quanh toàn bộ vùng ảnh
        rect = dlib.rectangle(0, 0, face_rgb.shape[1], face_rgb.shape[0])
        # Dự đoán landmarks
        landmarks = self.sp(face_rgb, rect)
        full_objs = dlib.full_object_detections()
        full_objs.append(landmarks)
        # Cắt và căn chỉnh face chip
        chips = dlib.get_face_chips(face_rgb, full_objs, size=size, padding=padding)
        return chips[0] if chips else None

def show_facechip():
    """
    Hiển thị face chip từ dataset FairFace để kiểm tra.
    """
    import pandas as pd
    import time

    df = pd.read_csv("data/fairface_label_val.csv")
    predictor = FaceAligner()

    try:
        for _, row in df.iterrows():
            img_file = os.path.join("data", row['file'])
            # Tải ảnh bằng OpenCV (BGR)
            img = cv2.imread(img_file)
            if img is None:
                continue
            start = time.time()
            facechip = predictor.detect_facechip(img)
            end = time.time()
            elapsed_time = end - start
            fps = 1 / elapsed_time
            print(f"FPS: {fps:.2f}")

            if facechip is not None:
                # Chuyển sang BGR để hiển thị bằng OpenCV
                facechip_bgr = cv2.cvtColor(facechip, cv2.COLOR_RGB2BGR)
                cv2.imshow('Detected Faces', facechip_bgr)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Đã bị gián đoạn bởi người dùng")
    finally:
        cv2.destroyAllWindows()

def detect_and_align(detector, aligner, image: np.ndarray, margin: float = 0.4, padding: float = 0.25):
    """
    Toàn bộ pipeline: phát hiện → mở rộng bbox → cắt ROI → căn chỉnh.

    Args:
        detector: Đối tượng phát hiện khuôn mặt (Mediapipe FaceDetection).
        aligner: Đối tượng FaceAligner.
        image (np.ndarray): Ảnh đầu vào định dạng BGR.
        margin (float): Độ mở rộng tương đối quanh bbox.
        padding (float): Độ mở rộng cho việc căn chỉnh.

    Returns:
        np.ndarray: Face chip đã căn chỉnh hoặc None.
    """
    infos, _ = detector.detect(image)
    if not infos:
        return None

    # Lấy bbox đầu tiên
    bbox = infos[0]['bbox']
    h, w = image.shape[:2]
    # Tính toán bbox tuyệt đối với margin
    x1 = max(0, int((bbox.xmin * w) - bbox.width * w * margin))
    y1 = max(0, int((bbox.ymin * h) - bbox.height * h * margin))
    x2 = min(w, int(x1 + bbox.width * w * (1 + 2 * margin)))
    y2 = min(h, int(y1 + bbox.height * h * (1 + 2 * margin)))

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    face_chip = aligner.aligning(roi, padding=padding)
    return face_chip

def natural_keys(text):
    """
    Hàm sắp xếp theo thứ tự tự nhiên (natural sort).

    Args:
        text (str): Chuỗi cần sắp xếp.

    Returns:
        list: Danh sách các thành phần để so sánh.
    """
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

def main():
    """
    Hàm chính để chạy thử pipeline phát hiện và căn chỉnh khuôn mặt.
    """
    import pandas as pd
    import time
    import glob
    import os
    import cv2
    from mdp_aligner import FaceDetection  # Sử dụng Mediapipe Face Detection

    detector = FaceDetection()
    aligner = FaceAligner()

    folder = "../../../output_frames_id/id_2"
    images_list = sorted(glob.glob(os.path.join(folder, '*.jpg')), key=natural_keys)

    total_time = 0
    frame_count = 0

    try:
        for imagepath in images_list:
            image = cv2.imread(imagepath)
            start = time.time()
            # Sử dụng detect_and_align với Mediapipe
            facechip = detect_and_align(detector, aligner, image)
            end = time.time()
            elapsed_time = end - start
            total_time += elapsed_time
            frame_count += 1

            fps = 1 / elapsed_time
            print(f"FPS (phát hiện và căn chỉnh): {fps:.2f}")

            if facechip is not None:
                # Chuyển từ RGB sang BGR để hiển thị
                facechip_bgr = cv2.cvtColor(facechip, cv2.COLOR_RGB2BGR)
                cv2.imshow('Detected Faces', facechip_bgr)

            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"\nFPS trung bình: {avg_fps:.2f}")

    except KeyboardInterrupt:
        print("Đã bị gián đoạn bởi người dùng")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()