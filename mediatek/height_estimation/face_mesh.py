import cv2
import mediapipe as mp
import numpy as np

class FaceMeshHeadSizeCalculator:
    # Chỉ số landmark cần thiết
    LEFT_EYE_LANDMARK = 33
    RIGHT_EYE_LANDMARK = 263
    MOUTH_LANDMARK = 13

    def __init__(self, static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Khởi tạo đối tượng FaceMesh với các tham số điều chỉnh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        # Tùy chọn DrawingSpec để vẽ landmarks
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def calculate_head_size(self, image_path: str) -> float:
        """
        Nhận đường dẫn ảnh, xử lý ảnh và tính head size theo công thức:
        
            head_size = distancePx(eyes_center, mouth) * 3
        
        Trả về kích thước đầu tính bằng pixel (float). Nếu không phát hiện được khuôn mặt, trả về -1.
        """
        image = cv2.imread(image_path)
        if image is None:
            print("Không thể đọc ảnh từ đường dẫn đã cho.")
            return -1

        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            # Sử dụng khuôn mặt đầu tiên nếu có nhiều khuôn mặt
            face_landmarks = results.multi_face_landmarks[0]

            left_eye = np.array([
                face_landmarks.landmark[self.LEFT_EYE_LANDMARK].x * w,
                face_landmarks.landmark[self.LEFT_EYE_LANDMARK].y * h
            ])
            right_eye = np.array([
                face_landmarks.landmark[self.RIGHT_EYE_LANDMARK].x * w,
                face_landmarks.landmark[self.RIGHT_EYE_LANDMARK].y * h
            ])
            mouth = np.array([
                face_landmarks.landmark[self.MOUTH_LANDMARK].x * w,
                face_landmarks.landmark[self.MOUTH_LANDMARK].y * h
            ])

            # Trung bình tọa độ của hai mắt
            eyes_center = (left_eye + right_eye) / 2

            # Tính khoảng cách Euclidean giữa mắt và miệng
            distance_px = np.linalg.norm(eyes_center - mouth)
            head_size = distance_px * 3
            return head_size
        else:
            print("Không phát hiện được khuôn mặt nào.")
            return -1

    def visualize_face_mesh(self, image_path: str, output_path: str = None):
        """
        Xử lý ảnh từ đường dẫn, vẽ các landmarks của FaceMesh và hiển thị kết quả.
        
        Nếu có khuôn mặt được phát hiện, tính head size và vẽ thêm thông tin lên ảnh.
        Nếu output_path được cung cấp, ảnh đã xử lý sẽ được lưu.
        """
        image = cv2.imread(image_path)
        if image is None:
            print("Không thể đọc ảnh từ đường dẫn đã cho.")
            return

        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Vẽ tất cả các landmark sử dụng Drawing Utils của mediapipe
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

                # Tính toán tọa độ pixel cho các điểm cần thiết
                left_eye = np.array([
                    face_landmarks.landmark[self.LEFT_EYE_LANDMARK].x * w,
                    face_landmarks.landmark[self.LEFT_EYE_LANDMARK].y * h
                ])
                right_eye = np.array([
                    face_landmarks.landmark[self.RIGHT_EYE_LANDMARK].x * w,
                    face_landmarks.landmark[self.RIGHT_EYE_LANDMARK].y * h
                ])
                mouth = np.array([
                    face_landmarks.landmark[self.MOUTH_LANDMARK].x * w,
                    face_landmarks.landmark[self.MOUTH_LANDMARK].y * h
                ])

                eyes_center = (left_eye + right_eye) / 2
                distance_px = np.linalg.norm(eyes_center - mouth)
                head_size = distance_px * 3

                # Vẽ các điểm đặc trưng
                cv2.circle(image, tuple(eyes_center.astype(int)), 5, (0, 255, 0), -1)  # Màu xanh lá cho trung tâm mắt
                cv2.circle(image, tuple(mouth.astype(int)), 5, (0, 0, 255), -1)        # Màu đỏ cho miệng

                # Ghi thông tin head size lên ảnh
                cv2.putText(image, f"Head Size: {head_size:.2f}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(image, "Khong phat hien khuon mat", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Hiển thị ảnh kết quả
        cv2.imshow("FaceMesh Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Lưu ảnh nếu output_path được cung cấp
        if output_path:
            cv2.imwrite(output_path, image)

    def close(self):
        """
        Giải phóng tài nguyên của FaceMesh.
        """
        self.face_mesh.close()


# Ví dụ sử dụng lớp
if __name__ == "__main__":
    calculator = FaceMeshHeadSizeCalculator()
    image_path = "face.jpg"   # Thay bằng đường dẫn đến ảnh khuôn mặt của bạn
    output_path = "output_face.jpg"   # Đường dẫn lưu ảnh kết quả sau khi vẽ
    
    # Tính head size
    head_size = calculator.calculate_head_size(image_path)
    if head_size != -1:
        print(f"Head size (pixels): {head_size:.2f}")
    
    # Visualize kết quả facemesh và lưu ảnh nếu cần
    calculator.visualize_face_mesh(image_path, output_path)
    calculator.close()
