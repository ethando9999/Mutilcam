import cv2
import numpy as np


import mediapipe as mp
print("Đường dẫn module mediapipe:", mp.__file__)
import time
from .dlib_aligner import FaceAligner


class FaceDetection:
    """
    Class for detecting faces in an image using MediaPipe Face Detection. 

    Attributes:
        model_selection (int): 0 for short-range, 1 for full-range detection.
        min_detection_confidence (float): Minimum confidence threshold for detection.
    """
    def __init__(self, model_selection: int = 0, min_detection_confidence: float = 0.8):  
        # Khởi tạo các module từ Mediapipe
        self.mp_face_detection = mp.solutions.face_detection  
        self.mp_drawing = mp.solutions.drawing_utils
        # Lưu ngưỡng confidence để lọc kết quả
        self.min_detection_confidence = min_detection_confidence
        # Khởi tạo mô hình phát hiện khuôn mặt
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        self.face_aligner = FaceAligner()
        self.fps_avg = 0.0
        self.call_count = 0

    def detect(self, image: np.ndarray):
        """
        Detect faces in the input BGR image.

        Args:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            detections_info (list of dict): Chỉ chứa detections có confidence ≥ min_detection_confidence
            detections (list): Raw MediaPipe Detection objects tương ứng.
        """
        start_time = time.time()
        # Chuyển đổi ảnh sang định dạng RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)

        detections_info = []
        detections = []

        if results.detections:
            for det in results.detections:
                score = det.score[0]
                # Lọc các detection theo ngưỡng confidence
                if score < self.min_detection_confidence:
                    continue

                info = {
                    'confidence': score,
                    'bbox': det.location_data.relative_bounding_box,
                    'keypoints': [(kp.x, kp.y) for kp in det.location_data.relative_keypoints]
                }
                detections_info.append(info)
                detections.append(det)

        # Tính FPS trung bình
        end_time = time.time()
        duration = end_time - start_time
        fps_current = 1 / duration if duration > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1
        # print(f"FPS Face detection: {self.fps_avg:.2f}")

        return detections_info, detections

    def get_square_roi_above(self,
                            image: np.ndarray,
                            bbox,
                            margin: float = 0.25) -> np.ndarray | None:
        """
        Trích xuất ROI hình vuông, mở rộng lên trên khỏi bbox nhưng
        luôn nằm gọn trong ảnh. Trả về None nếu không còn pixel hợp lệ.

        Args:
            image (np.ndarray): Ảnh gốc (BGR).
            bbox:  Đối tượng có (xmin, ymin, width, height) hoặc
                (xmin, ymin, xmax, ymax) – các giá trị chuẩn hoá 0–1.
            margin (float): Tỷ lệ nới bbox trước khi cắt.

        Returns:
            np.ndarray | None: ROI hình vuông (BGR) hoặc None.
        """
        h, w = image.shape[:2]

        # Lấy (xmin, ymin, xmax, ymax) chuẩn hoá
        x_min = bbox.xmin
        y_min = bbox.ymin
        if hasattr(bbox, "xmax"):            # đã có xmax/ymax
            x_max = bbox.xmax
            y_max = bbox.ymax
        else:                                # kiểu width/height
            x_max = bbox.xmin + bbox.width
            y_max = bbox.ymin + bbox.height

        # Quy ra pixel & tính cạnh mở rộng
        x_center = (x_min + x_max) / 2 * w
        bbox_w   = (x_max - x_min) * w
        bbox_h   = (y_max - y_min) * h
        side     = int(max(bbox_w, bbox_h) * (1 + 2 * margin))

        # Giới hạn side không lớn hơn ảnh
        side = max(1, min(side, w, h))

        # Toạ độ tạm
        y2 = int(y_max * h)          # đáy bbox gốc
        y1 = y2 - side               # mở rộng lên trên
        if y1 < 0:                   # đụng nóc → thu cạnh
            side = y2                # (vì y1 sẽ thành 0)
            y1  = 0
            side = max(1, min(side, w))  # vẫn đảm bảo ≤ w
        # Cập nhật lại toạ độ theo side mới
        x1 = int(x_center - side / 2)
        x2 = x1 + side

        # Nếu tràn trái/phải → tịnh tiến để vẫn giữ hình vuông
        if x1 < 0:
            x1, x2 = 0, side
        elif x2 > w:
            x2, x1 = w, w - side

        # Lấy ROI cuối cùng
        y2 = y1 + side               # y2 có thể thay đổi nếu side thay
        roi = image[y1:y2, x1:x2]

        # Kiểm tra lần cuối
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return None
        # Cắt vuông tuyệt đối (nếu còn lệch 1 px do làm tròn)
        side_final = min(roi.shape[:2])
        return roi[:side_final, :side_final]

    def draw_detections(self, image: np.ndarray, detections: list): 
        """
        Draw detected faces on the image.

        Args:
            image (numpy.ndarray): Input image in BGR format.
            detections (list): Raw MediaPipe Detection objects to draw.

        Returns:
            annotated_image (numpy.ndarray): Copy of input with drawn detections. 
        """
        annotated_image = image.copy()
        if detections:
            for det in detections:
                self.mp_drawing.draw_detection(annotated_image, det) 
        return annotated_image

    def detect_and_align(self, image, margin: float = 0.3, padding: float = 0.2):
        """
        Detect and align faces in the image.

        Args:
            image (numpy.ndarray): Input image in BGR format.
            margin (float): Margin around the detected face bounding box.
            padding (float): Padding for face alignment.

        Returns:
            aligned_face (numpy.ndarray): Aligned face image, or None if no face is detected.
        """
        infos, _ = self.detect(image)
        if not infos:
            return None

        bbox = infos[0]['bbox']
        h, w = image.shape[:2]
        x1 = max(0, int((bbox.xmin * w) - bbox.width * w * margin))
        y1 = max(0, int((bbox.ymin * h) - bbox.height * h * margin))
        x2 = min(w, int(x1 + bbox.width * w * (1 + 2 * margin)))
        y2 = min(h, int(y1 + bbox.height * h * (1 + 2 * margin)))
        roi = image[y1:y2, x1:x2]

        # roi = self.get_square_roi_above(image, bbox, margin=0.3) 
        if roi.size == 0: 
            return None

        return self.face_aligner.aligning(roi, padding=padding)

def natural_keys(text):
    """
    Hàm sắp xếp theo thứ tự tự nhiên (natural sort)

    Args:
        text (str): Chuỗi cần sắp xếp.

    Returns:
        list: Danh sách các phần tử đã tách để sắp xếp tự nhiên.
    """
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

# Ví dụ sử dụng
if __name__ == '__main__':
    import os
    import glob
    detector = FaceDetection()
    # folder = "../data/val"  # Thay đổi đường dẫn thư mục nếu cần
    folder = "../../../output_frames_id/id_3"
    if os.path.exists(folder):
        print(f"Folder tồn tại: {folder}")
    else:
        print(f"Folder không tồn tại: {folder}")
    images_list = sorted(glob.glob(os.path.join(folder, '*.jpg')), key=natural_keys)
    if images_list is not None:
        print(f"Số lượng ảnh: {len(images_list)}")
    else:
        print("Không có ảnh nào được tìm thấy")
    try:
        invalid_count = 0  # Số ảnh không hợp lệ

        for imagepath in images_list:
            image = cv2.imread(imagepath)
            facechip = detector.detect_and_align(image)

            if facechip is None or facechip.size == 0:
                print(f"⚠️ Bỏ qua ảnh không có face hợp lệ: {imagepath}")
                invalid_count += 1
                continue

            cv2.imshow('Detected Faces', facechip)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

        print(f"Tổng số ảnh không hợp lệ: {invalid_count}")

    except KeyboardInterrupt:
        print("Đã bị gián đoạn bởi người dùng")
    finally: 
        cv2.destroyAllWindows()