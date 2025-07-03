import os
import cv2
import numpy as np
import time
from collections import Counter
import torch
from feature import feature_extraction
from pose_color_signature import PoseColorSignatureExtractor
from reid_manager import ReIDManager

class PersonReID:
    def __init__(self):
        # Khởi tạo các thông số
        self.OUTPUT_DIR = 'server_python/output_frames_id'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Khởi tạo device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Khởi tạo các model và manager
        self.signature_extractor = PoseColorSignatureExtractor(
            n_cluster=3,
            d=3,
            n_iter=100,
            verbose=False
        )
        self.feature_extractor = feature_extraction.FeatureModel(device=self.device)
        self.reid_manager = ReIDManager(
            similarity_threshold=0.7,
            iou_threshold=0.1
        )
        
        # Khởi tạo biến đếm frame và thống kê
        self.frame_index = 0
        self.id_statistics = Counter()
        self.update_interval = 90  # 3 giây với 30fps

    def process_frame(self, frame_data):
        frame = frame_data["frame"]
        boxes_data = frame_data["boxes_data"]
        keypoints_data = frame_data["keypoints_data"]
        
        if frame is None or len(boxes_data) == 0:
            return frame

        start_time = time.time()

        for i, box in enumerate(boxes_data):
            x1, y1, x2, y2 = map(int, box)

            # Crop ảnh người trực tiếp từ frame gốc
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # Tính feature
            features_person = self.feature_extractor.extract_features(person_img)

            # Tính color signature
            body_signature = None
            if keypoints_data is not None and len(keypoints_data) > i:
                kpts = keypoints_data[i]
                edge_pixels = self.signature_extractor.extract_edge_pixels(frame, kpts)
                body_signature = self.signature_extractor.get_body_color_signature(edge_pixels)

            # Gán ID
            bbox_original = (x1, y1, x2, y2)
            final_id = self.reid_manager.update_or_create_track(
                features_person,
                body_signature,
                bbox_original
            )

            # Vẽ khung + ID lên frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {final_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Lưu crop vào folder của ID
            id_folder = os.path.join(self.OUTPUT_DIR, f"id_{final_id}")
            os.makedirs(id_folder, exist_ok=True)
            crop_filename = f"frame_{self.frame_index}_det{i}.jpg"
            crop_path = os.path.join(id_folder, crop_filename)
            cv2.imwrite(crop_path, person_img)

            # Thống kê
            self.id_statistics[final_id] += 1

        # Lưu frame đã vẽ bbox và ID
        frame_filename = f"frame_{self.frame_index}_tracked.jpg"
        frame_save_path = os.path.join(self.OUTPUT_DIR, frame_filename)
        cv2.imwrite(frame_save_path, frame)

        # In thống kê mỗi 3 giây
        if self.frame_index % self.update_interval == 0 and self.frame_index > 0:
            print("\nID Statistics:")
            for id_stat in self.id_statistics.most_common():
                print(f"ID: {id_stat[0]}, Count: {id_stat[1]}")
            self.id_statistics.clear()

        # Tính và in FPS
        processing_time = time.time() - start_time
        current_fps = 1 / processing_time if processing_time > 0 else 0
        print(f"Frame {self.frame_index}: FPS = {current_fps:.2f}")

        self.frame_index += 1
        return frame

# Hàm process_frame chính để xử lý async
person_reid = PersonReID()

async def process_frame(frame_queue):
    while True:
        frame_data = await frame_queue.get()
        processed_frame = person_reid.process_frame(frame_data)
        # Có thể thêm xử lý khác với processed_frame ở đây nếu cần