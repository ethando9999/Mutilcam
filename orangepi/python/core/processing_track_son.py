# file: python/core/processing_RGBD.py (Cập nhật khởi tạo và xử lý kết quả)

import asyncio
import cv2
import os
from datetime import datetime
import numpy as np
import time
from typing import Optional, Dict
from collections import Counter 

# Giả định các module này tồn tại và hoạt động đúng
from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector import StereoProjector
from .keypoints_handle import get_head_center, get_torso_box, adjust_keypoints_to_box
from .height_estimator import HeightEstimator
import config

# ======================= PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =======================
# --- TÍCH HỢP CÁC MODULE PHÂN TÍCH ---
from color.pose_color_signature_new import PoseColorAnalyzer
from color.clothing_classifier_son import ClothingClassifier
from gender_yolo import GenderRecognition
# ===================== KẾT THÚC PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =====================

logger = get_logger(__name__)

device_id = config.OPI_CONFIG.get("device_id", "opi_01")

class FrameProcessor:
    """
    Xử lý các lô frame RGB-D, từ phát hiện người cho đến trích xuất thông tin chi tiết.
    Tích hợp logic gửi dữ liệu qua WebSocket một cách thông minh và làm mịn dữ liệu.
    """
    def __init__(self, calib_path: str, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue, batch_size: int = 2):
        """ 
        Khởi tạo tất cả các module cần thiết và các biến trạng thái.
        """
        self.detector = HumanDetection()
        self.stereo_projector = StereoProjector(calib_file_path=calib_path)
        
        mtx_rgb = self.stereo_projector.params.get('mtx_rgb')
        if mtx_rgb is None: raise ValueError("mtx_rgb không có trong file hiệu chỉnh.")
        self.height_estimator = HeightEstimator(mtx_rgb)

        # <<< THAY ĐỔI: Cập nhật khởi tạo các module theo logic mới >>>
        
        # Gender recognizer (giữ nguyên)
        gender_model_path = config.OPI_CONFIG["GENDER_MODEL_PATH"]
        self.gender_recognizer = GenderRecognition(model_path=gender_model_path)
        
        # ClothingClassifier với logic mới - sử dụng skin CSV và các ngưỡng mới
        skin_csv_path = config.OPI_CONFIG["SKIN_TONE_CSV_PATH"]
        self.clothing_classifier = ClothingClassifier(
            skin_csv_path=skin_csv_path,
            sleeve_color_similarity_threshold=config.OPI_CONFIG.get("SLEEVE_COLOR_THRESHOLD", 10.0),
            pants_color_similarity_threshold=config.OPI_CONFIG.get("PANTS_COLOR_THRESHOLD", 40.0)
        )

        # PoseColorAnalyzer đơn giản hóa 
        self.color_extractor = PoseColorAnalyzer(
            line_thickness=config.OPI_CONFIG.get("LINE_THICKNESS", 30)
        )
        
        self.analysis_output_dir = os.path.join(config.OPI_CONFIG.get("results_dir", "results"), "analysis_panels")
        os.makedirs(self.analysis_output_dir, exist_ok=True)

        self.people_count_queue = people_count_queue
        self.height_queue = height_queue
        
        self.semaphore = asyncio.Semaphore(4)
        self.batch_size = batch_size
        
        self.debug_dir = os.path.join(config.OPI_CONFIG.get("results_dir", "results"), "debug_projection")
        os.makedirs(self.debug_dir, exist_ok=True)

        self.temp_count_queue = asyncio.Queue()
        self.count_buffer_size = 3

        self.pass_count = -1
        self.fps_avg = 0.0
        self.call_count = 0    
        self.table_id = config.OPI_CONFIG.get("SOCKET_TABLE_ID", 1)
        self.send_zero_flag = False

        logger.info("FrameProcessor (Cập nhật logic mới cho phân tích trang phục) đã được khởi tạo.")
    
    # --------------------------------------------------------------------
    # CÁC HÀM PHỤ TRỢ (HELPERS) - Giữ nguyên
    # --------------------------------------------------------------------
    
    async def _get_clothing_analysis_async(self, rgb_frame: np.ndarray, keypoints: np.ndarray) -> Optional[Dict]:
        """
        Pipeline phân tích quần áo: gọi module xử lý đầy đủ và trả về kết quả.
        """
        try:
            # Gọi hàm xử lý đầy đủ và trả về kết quả trực tiếp.
            # Việc phân loại (classification) đã được thực hiện BÊN TRONG hàm này rồi.
            analysis_result = await self.color_extractor.process_and_classify(
                rgb_frame, 
                keypoints,
                self.clothing_classifier # Truyền classifier vào để nó sử dụng
            )
            return analysis_result
            
        except Exception as e: 
            logger.error(f"Lỗi trong pipeline phân tích trang phục: {e}", exc_info=True) 
            return None

    async def _create_analysis_panel_async(self, person_image: np.ndarray, attributes: dict, filename: str):
        """[CẬP NHẬT] Tạo ảnh panel theo format kết quả mới."""
        try:
            gender_analysis = attributes.get("gender_analysis", {})
            clothing_analysis = attributes.get("clothing_analysis", {})
            classification = clothing_analysis.get("classification", {})
            raw_colors = clothing_analysis.get("raw_color_data", {})

            gender_label = gender_analysis.get('gender', 'N/A').capitalize()
            
            # <<< THAY ĐỔI: Cập nhật theo format kết quả mới >>>
            sleeve_type = classification.get("sleeve_type", "KHONG THE XAC DINH")
            pants_type = classification.get("pants_type", "KHONG THE XAC DINH")
            skin_tone_id = classification.get("skin_tone_id")
            skin_tone_bgr = classification.get("skin_tone_bgr")

            # Tạo panel hiển thị
            img_h, img_w = person_image.shape[:2]
            panel_w = 400
            summary_h = max(img_h, 400)
            summary_img = np.full((summary_h, img_w + panel_w, 3), (255, 255, 255), dtype=np.uint8)
            summary_img[0:img_h, 0:img_w] = person_image

            text_x, text_color = img_w + 20, (0, 0, 0)
            y_pos = 30
            def draw_text(text, y, font_scale=0.6, thickness=1):
                cv2.putText(summary_img, text, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                return y + 30

            y_pos = draw_text("Ket Qua Phan Tich", y_pos, 0.7, 2)
            y_pos = draw_text(f"Gioi Tinh: {gender_label}", y_pos)
            
            # Hiển thị kết quả phân loại mới
            y_pos = draw_text(f"Ao: {sleeve_type}", y_pos, 0.6, 2)
            y_pos = draw_text(f"Quan: {pants_type}", y_pos, 0.6, 2)
            
            # Hiển thị tone màu da nếu có
            if sleeve_type == "AO NGAN TAY" and skin_tone_id and skin_tone_bgr:
                y_pos = draw_text("Tone Da:", y_pos, 0.6, 2)
                # Vẽ ô màu
                cv2.rectangle(summary_img, (text_x, y_pos - 25), (text_x + 30, y_pos - 5), tuple(map(int, skin_tone_bgr)), -1)
                cv2.rectangle(summary_img, (text_x, y_pos - 25), (text_x + 30, y_pos - 5), (0, 0, 0), 1)
                draw_text(f"Tone #{skin_tone_id}", y_pos, 0.5)
                y_pos += 25

            # Hiển thị màu raw data nếu cần
            display_data = {
                "Mau Ao": raw_colors.get("torso_colors"),
                "Mau Tay": raw_colors.get("forearm_colors"),
                "Mau Dui": raw_colors.get("thigh_colors"),
                "Mau Ong": raw_colors.get("shin_colors"),
            }
            
            for label, colors_data in display_data.items():
                if colors_data and colors_data[0]:  # Chỉ có 1 màu trong logic mới
                    color_info = colors_data[0]
                    bgr = color_info["bgr"]
                    y_pos = draw_text(f"{label}:", y_pos, 0.5, 1)
                    cv2.rectangle(summary_img, (text_x + 15, y_pos - 20), (text_x + 35, y_pos - 5), tuple(map(int, bgr)), -1)
                    draw_text(f"BGR: {bgr}", y_pos, 0.4)
                    y_pos += 20
            
            output_path = os.path.join(self.analysis_output_dir, filename)
            await self._run_in_executor(cv2.imwrite, output_path, summary_img)

        except Exception as e:
            logger.error(f"Không thể tạo panel '{filename}': {e}", exc_info=True)
    
    # Các hàm helper khác giữ nguyên
    def _is_detection_valid(self, box: tuple, keypoints: np.ndarray, frame_shape: tuple) -> bool:
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        if np.any(np.isnan(box)) or np.any(np.isinf(box)): return False
        if x1 >= x2 or y1 >= y2: return False
        if x2 < 0 or y2 < 0 or x1 > w or y1 > h: return False
        
        if keypoints.shape[1] >= 3:
            valid_kpts_count = np.sum(keypoints[:, 2] > 0.3)
        else:
            valid_kpts_count = np.sum(np.sum(keypoints[:, :2], axis=1) > 0)
        
        if valid_kpts_count < 4:
            return False
        return True

    async def _run_in_executor(self, func, *args):
        return await asyncio.to_thread(func, *args)

    async def save_debug_images(self, base_name, rgb_frame, tof_depth_frame, rgb_box, tof_box):
        try:
            tof_color = cv2.normalize(tof_depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            tof_color_map = cv2.applyColorMap(tof_color, cv2.COLORMAP_JET)
            x1, y1, x2, y2 = rgb_box
            rgb_crop = rgb_frame[y1:y2, x1:x2]
            
            if rgb_crop.size > 0: await self._run_in_executor(cv2.imwrite, os.path.join(self.debug_dir, f"{base_name}_crop_rgb.png"), rgb_crop)
            if tof_box:
                tx1, ty1, tx2, ty2 = tof_box
                tof_crop = tof_color_map[ty1:ty2, tx1:tx2]
                if tof_crop.size > 0: await self._run_in_executor(cv2.imwrite, os.path.join(self.debug_dir, f"{base_name}_crop_tof.png"), tof_crop)
                cv2.rectangle(tof_color_map, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            await self._run_in_executor(cv2.imwrite, os.path.join(self.debug_dir, f"{base_name}_full_rgb_boxed.png"), rgb_frame)
            await self._run_in_executor(cv2.imwrite, os.path.join(self.debug_dir, f"{base_name}_full_tof_boxed.png"), tof_color_map)
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh debug: {e}")
            
    def _get_feet_point(self, keypoints: np.ndarray) -> Optional[tuple[int, int]]:
        """
        Lấy điểm chân (điểm thấp nhất) từ các keypoints.
        Ưu tiên các điểm mắt cá chân (ankle) có confidence cao.
        YOLOv8-Pose keypoints: 15: left_ankle, 16: right_ankle.
        """
        if keypoints.shape[0] <= 16: return None # Đảm bảo có đủ keypoints
        
        ankle_kpts = keypoints[[15, 16], :]
        
        # Lọc các keypoint có confidence > 0.3 (một ngưỡng hợp lý)
        if keypoints.shape[1] >= 3:
            visible_ankles = ankle_kpts[ankle_kpts[:, 2] > 0.3]
        else: # Dự phòng nếu không có confidence
            visible_ankles = ankle_kpts[np.sum(ankle_kpts, axis=1) > 0]
            
        if len(visible_ankles) == 0:
            return None
        
        # Tìm điểm chân thấp nhất trên ảnh (có tọa độ v lớn nhất)
        lowest_foot = visible_ankles[np.argmax(visible_ankles[:, 1])]
        
        return int(lowest_foot[0]), int(lowest_foot[1])

    # --------------------------------------------------------------------
    # HÀM XỬ LÝ CHÍNH (CORE PROCESSING) - Phần lớn giữ nguyên
    # --------------------------------------------------------------------

    async def process_human_async(self, frame_id: int, rgb_frame: np.ndarray, tof_depth_map: np.ndarray, box: tuple, keypoints: np.ndarray):
        """
        Luồng xử lý tối ưu cho một người. Gọi hàm cấp cao để lấy tọa độ và khoảng cách.
        """
        async with self.semaphore:
            try:
                # BƯỚC 1: LẤY TỌA ĐỘ SÀN VÀ KHOẢNG CÁCH CÙNG LÚC
                torso_box = get_torso_box(keypoints, box[:4])
                feet_point = self._get_feet_point(keypoints)
                cam_angle = config.OPI_CONFIG.get("CAM_ANGLE_DEG")

                world_data = await self._run_in_executor(
                    self.stereo_projector.get_worldpoint_coordinates,
                    torso_box, feet_point, tof_depth_map, cam_angle
                )

                if world_data is None:
                    return None # Lỗi đã được log bên trong hàm
                
                distance_mm = world_data["distance_mm"]
                world_point_xy = world_data["floor_pos_cm"]

                # Làm tròn và giới hạn tọa độ: x ∈ [-150, 150], y ∈ [0, 300]
                x = max(-150, min(150, int(world_point_xy[0])))
                y = max(0, min(300, int(world_point_xy[1])))

                world_point_xy = (x, y)

                logger.info(f"✅ Người hợp lệ. Khoảng cách: {distance_mm/1000:.2f}m. Bắt đầu xử lý sâu...")
                distance_m = distance_mm / 1000.0

                # BƯỚC 2: CHẠY SONG SONG CÁC TÁC VỤ CÒN LẠI
                height_task = self._run_in_executor(self.height_estimator.estimate, keypoints, distance_m)
                
                human_box_img = rgb_frame[box[1]:box[3], box[0]:box[2]]
                if human_box_img.size == 0: return None
                
                adjusted_keypoints = adjust_keypoints_to_box(keypoints, box[:4])
                body_parts_task = self._run_in_executor(extract_body_parts_from_frame, human_box_img, adjusted_keypoints)
                projection_task = self._run_in_executor(self.stereo_projector.project_rgb_box_to_tof, box[:4], distance_mm, tof_depth_map.shape)
                gender_task = self._run_in_executor(self.gender_recognizer.predict, human_box_img)
                clothing_task = self._get_clothing_analysis_async(human_box_img, adjusted_keypoints)
                
                (est_height_m, height_status), body_parts_boxes, tof_box_projected, gender_analysis, clothing_analysis = await asyncio.gather(
                    height_task, body_parts_task, projection_task, gender_task, clothing_task,
                    return_exceptions=True # Xử lý lỗi riêng lẻ
                )
                
                if isinstance(gender_analysis, Exception):
                    logger.error(f"Lỗi phân tích giới tính: {gender_analysis}")
                    gender_analysis = {} # Đặt về dict rỗng
                if isinstance(clothing_analysis, Exception):
                    logger.error(f"Lỗi phân tích quần áo: {clothing_analysis}")
                    clothing_analysis = {} # Đặt về dict rỗng
                if isinstance(est_height_m, Exception): est_height_m = None
                if isinstance(height_status, Exception): height_status = "Error"

                if tof_box_projected:
                    debug_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}_F{frame_id}"
                    asyncio.create_task(self.save_debug_images(debug_base_name, rgb_frame.copy(), tof_depth_map.copy(), box[:4], tof_box_projected))
                
                # <<< CẬP NHẬT: Log kết quả theo format mới >>>
                clothing_classification = clothing_analysis.get("classification", {}) if clothing_analysis else {}
                sleeve_info = clothing_classification.get("sleeve_type", "N/A")
                pants_info = clothing_classification.get("pants_type", "N/A")
                skin_tone_info = f"#{clothing_classification.get('skin_tone_id')}" if clothing_classification.get('skin_tone_id') else "N/A"
                
                logger.info(
                    f"✅✅ Xử lý hoàn tất: Khoảng cách={distance_m:.2f}m, "
                    f"Chiều cao={(f'{est_height_m:.2f}m' if est_height_m is not None else 'None')} ({height_status}), "
                    f"Giới tính={gender_analysis.get('gender', 'N/A') if gender_analysis else 'N/A'}, "
                    f"Áo={sleeve_info}, Quần={pants_info}, Tone da={skin_tone_info}, "
                    f"Vị trí sàn={('({:.1f}, {:.1f}) cm'.format(*world_point_xy)) if world_point_xy else 'Không có'}"
                )
                
                attributes = {
                    "gender_analysis": gender_analysis if not isinstance(gender_analysis, Exception) else {},
                    "clothing_analysis": clothing_analysis if not isinstance(clothing_analysis, Exception) else {},
                }
                
                result_data = {
                    "frame_id": frame_id, 
                    "human_box_img": human_box_img,
                    "body_parts": body_parts_boxes,
                    "world_point_xy": world_point_xy,
                    "bbox": box, 
                    "map_keypoints": keypoints,
                    "distance_mm": distance_mm, 
                    "est_height_m": est_height_m,  
                    "height_status": height_status,
                    "time_detect": datetime.now().isoformat(),
                    "attributes": attributes
                }
                
                # Tạo panel phân tích một cách bất đồng bộ
                panel_filename = f"analysis_{device_id}_{frame_id}_{int(time.time())}.jpg"
                asyncio.create_task(self._create_analysis_panel_async(
                    result_data["human_box_img"],
                    result_data["attributes"],
                    panel_filename
                ))

                return result_data
            
            except Exception as e:
                logger.error(f"Lỗi xử lý người cho frame {frame_id}: {e}", exc_info=True)
                return None

    # Các hàm còn lại giữ nguyên...
    async def _manage_people_count_state(self, current_person_count: int):
        """
        Quản lý bộ đệm đếm người và chỉ gửi đi giá trị ổn định nhất.
        """
        await self.temp_count_queue.put(current_person_count)

        if self.temp_count_queue.qsize() >= self.count_buffer_size:
            counts_in_buffer = []
            for _ in range(self.count_buffer_size):
                try:
                    count = self.temp_count_queue.get_nowait()
                    counts_in_buffer.append(count)
                except asyncio.QueueEmpty:
                    break
            
            if not counts_in_buffer:
                return

            stable_count = Counter(counts_in_buffer).most_common(1)[0][0]
            logger.info(f"Đã xử lý bộ đệm đếm: {counts_in_buffer} -> Gửi đi số lượng ổn định: {stable_count}")

            packet = {"total_person": stable_count}
            asyncio.create_task(self.people_count_queue.put(packet))

    # Các hàm process_frame_queue, enqueue_height và start_processor giữ nguyên...
    async def process_frame_queue(self, frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
        """Vòng lặp chính: Lấy dữ liệu, điều phối tác vụ và gửi kết quả một cách tối ưu."""
        frame_number = 0
        while True:
            start_time = time.time()
            batch_data_paths = []
            try:
                item = await asyncio.wait_for(frame_queue.get(), timeout=5.0)
                if item is None:
                    break
                batch_data_paths.append(item)
                frame_queue.task_done()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            while len(batch_data_paths) < self.batch_size:
                try:
                    item = frame_queue.get_nowait()
                    if item is None:
                        frame_queue.put_nowait(None)
                        break
                    batch_data_paths.append(item)
                    frame_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            try:
                # BƯỚC 1: Load ảnh RGB và depth
                loaded_data_map = {}
                frames_for_detection = []
                valid_indices = []

                for idx, (rgb_path, depth_path, amp_path) in enumerate(batch_data_paths):
                    try:
                        rgb_frame = cv2.imread(rgb_path)
                        depth_frame = np.load(depth_path)
                        if rgb_frame is not None and depth_frame is not None:
                            frames_for_detection.append(rgb_frame)
                            valid_indices.append(idx)
                            loaded_data_map[idx] = (rgb_frame, depth_frame)
                    except Exception as e:
                        logger.error(f"Lỗi đọc file {rgb_path}: {e}")

                if not frames_for_detection:
                    continue

                # BƯỚC 2: Phát hiện người (song song)
                detection_results = await asyncio.gather(
                    *[self._run_in_executor(self.detector.run_detection, f) for f in frames_for_detection],
                    return_exceptions=True
                )

                # BƯỚC 3: Tạo tác vụ xử lý sâu
                all_human_tasks = []
                people_count_per_frame = []

                for res_idx, res in enumerate(detection_results):
                    orig_idx = valid_indices[res_idx]
                    if not res or isinstance(res, Exception):
                        people_count_per_frame.append(0)
                        continue

                    kpts_data, boxes_data = res
                    rgb, depth = loaded_data_map[orig_idx]

                    valid_detections_in_frame = 0
                    for kpts, box in zip(kpts_data, boxes_data):
                        if self._is_detection_valid(box[:4], kpts, rgb.shape):
                            valid_detections_in_frame += 1
                            task = self.process_human_async(
                                frame_number + orig_idx, rgb, depth, box, kpts
                            )
                            all_human_tasks.append(task)

                    people_count_per_frame.append(valid_detections_in_frame)

                # BƯỚC 4: Gọi xử lý người và đóng gói packet
                final_results = await asyncio.gather(*all_human_tasks, return_exceptions=True)
                valid_results = [
                    res for res in final_results
                    if res and not isinstance(res, Exception)
                ]

                heights_cm = [
                    res["est_height_m"] * 100.0
                    for res in valid_results
                    if res.get("est_height_m") is not None
                ]

                if heights_cm:
                    # Có ít nhất một người → gửi danh sách các chiều cao
                    height_packet = {
                        "table_id": self.table_id,
                        "heights_cm": heights_cm
                    }
                else:
                    # Không có người → gửi giá trị 0
                    height_packet = {
                        "table_id": self.table_id,
                        "heights_cm": [0]
                }
                # luôn schedule task enqueue, không chờ
                asyncio.create_task(self.enqueue_height(height_packet))

                # Tạo danh sách người với các trường cần thiết
                people_list = [
                    {
                        "world_point_xy": res["world_point_xy"],
                        "bbox": res["bbox"],
                        "distance_mm": res["distance_mm"],
                        "est_height_m": res["est_height_m"],
                        "height_status": res["height_status"], 
                        "attributes": res["attributes"],
                    }
                    for res in valid_results
                ]

                # Đóng gói packet cuối cùng
                packet = {
                    "frame_id": frame_number,
                    "time_detect": datetime.now().isoformat(),
                    "people_list": people_list,
                    "camera_id": device_id,
                }

                # Đưa packet đã được tối ưu vào hàng đợi xử lý tracking
                await processing_queue.put(packet)
                logger.info(f"Đã put packet với {len(valid_results)} người vào processing_queue")

                # BƯỚC 5: Cập nhật FPS và frame_number
                frame_number += len(frames_for_detection)
                end_time = time.time()
                duration = end_time - start_time
                if duration > 0:
                    fps_current = len(batch_data_paths) / duration
                    self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
                    logger.info("FPS Avg Processing: %.2f", self.fps_avg)
                    self.call_count += 1
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong xử lý lô: {e}", exc_info=True)
                
    async def enqueue_height(self, height_packet):
            # nếu queue full thì loại bỏ phần tử cũ nhất
            if self.height_queue.full():
                try:
                    # nhanh gọn, không đợi
                    self.height_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            # giờ chắc chắn còn chỗ, put vào
            await self.height_queue.put(height_packet)
            logger.info(f"Da put vao height_queue: {height_packet}")
            
# --------------------------------------------------------------------
# HÀM KHỞI TẠO WORKER
# --------------------------------------------------------------------
async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue, calib_path: str):
    """
    Khởi tạo và chạy FrameProcessor worker.
    """
    logger.info("Khởi động worker xử lý...")
    try:
        processor = FrameProcessor( 
            calib_path=calib_path,
            people_count_queue=people_count_queue,
            height_queue=height_queue,
            batch_size=1
        )
        await processor.process_frame_queue(frame_queue, processing_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True)