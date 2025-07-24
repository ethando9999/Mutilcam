# file: python/core/processing_RGBD.py

import asyncio
import cv2
import os
from datetime import datetime
import numpy as np
import time
from typing import Optional,Dict
from collections import Counter 
import json
# Giả định các module này tồn tại và hoạt động đúng
from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector import StereoProjector
from .keypoints_handle import get_head_center, get_torso_box, adjust_keypoints_to_box, get_optimal_feet_point
from .height_estimator import HeightEstimator
import config

# ======================= PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =======================
# --- TÍCH HỢP CÁC MODULE PHÂN TÍCH ---
from pose_color_signature import PoseColorAnalyzer
from clothing_classifier import ClothingClassifier
from gender_yolo import GenderRecognition 
# ===================== KẾT THÚC PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =====================

logger = get_logger(__name__)

device_id = config.OPI_CONFIG.get("device_id", "opi_01")

class FrameProcessor:
    """
    [PHIÊN BẢN TỐI ƯU HOÀN CHỈNH - 23/07]
    - Sửa lỗi AttributeError do thiếu khởi tạo temp_count_queue.
    - Tích hợp logic xử lý batch và đếm người chi tiết.
    - Logic xử lý đồng bộ và hiệu quả.
    """
    def __init__(
        self,
        calib_path: str,
        people_count_queue: asyncio.Queue,
        height_queue: asyncio.Queue,
        processing_queue: asyncio.Queue,
        batch_size: int = 1
    ):
        # --- 1. Khởi tạo các module ---
        self.detector = HumanDetection()
        self.stereo_projector = StereoProjector(calib_file_path=calib_path)
        mtx_rgb = self.stereo_projector.params.get('mtx_rgb')
        if mtx_rgb is None: raise ValueError("mtx_rgb không có trong file hiệu chỉnh.")
        self.height_estimator = HeightEstimator(mtx_rgb)

        self.color_extractor = PoseColorAnalyzer(k_per_region=5, merge_threshold=25.0)
        
        skin_csv_path = config.OPI_CONFIG.get("SKIN_TONE_CSV_PATH")
        self.clothing_classifier = ClothingClassifier(
            skin_csv_path=skin_csv_path,
            skin_color_threshold=38.0,
            clothing_color_similarity_threshold=35.0
        )
        self.gender_model = GenderRecognition(
            model_path=config.OPI_CONFIG.get("GENDER_MODEL_PATH"),
            confidence_threshold=config.OPI_CONFIG.get("GENDER_CONFIDENCE_THRESHOLD")
        )

        # --- 2. Khởi tạo Queues, Semaphore và các biến trạng thái ---
        self.people_count_queue = people_count_queue
        self.height_queue = height_queue
        self.semaphore = asyncio.Semaphore(4)
        self.batch_size = batch_size
        self.table_id = config.OPI_CONFIG.get("SOCKET_TABLE_ID")
        self.fps_avg = 0.0
        self.call_count = 0
        
        # [SỬA LỖI] Khởi tạo các thuộc tính cho việc đếm người
        self.temp_count_queue = asyncio.Queue()
        self.count_buffer_size = 3
        
        logger.info(f"✅ FrameProcessor đã được khởi tạo.")

    async def _run_in_executor(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
        
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

    async def _create_analysis_panel_async(self, person_image: np.ndarray, attributes: dict, filename: str):
        """[SỬA LỖI & CẬP NHẬT] Tạo ảnh panel từ gói thuộc tính mới."""
        try:
            gender_analysis = attributes.get("gender_analysis", {})
            clothing_analysis = attributes.get("clothing_analysis", {})
            classification = clothing_analysis.get("classification", {})
            raw_colors = clothing_analysis.get("raw_color_data", {})

            gender_label = gender_analysis.get('gender', 'N/A').capitalize()
            sleeve_type = classification.get("sleeve_type", "N/A")
            pants_type = classification.get("pants_type", "N/A")

            display_data = {
                f"Loai Ao: {sleeve_type}": raw_colors.get("torso_colors"),
                f"Loai Quan: {pants_type}": raw_colors.get("thigh_colors"),
            } 

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
            
            for label, colors_data in display_data.items():
                y_pos = draw_text(label, y_pos, 0.6, 2)
                if colors_data:
                    for color_info in sorted(colors_data, key=lambda x: x["percentage"], reverse=True)[:3]:
                        bgr, pct = color_info["bgr"], color_info["percentage"]
                        cv2.rectangle(summary_img, (text_x + 15, y_pos), (text_x + 45, y_pos + 20), tuple(map(int, bgr)), -1)
                        draw_text(f"BGR: {bgr}, {pct:.1f}%", y_pos + 16, 0.5)
                        y_pos += 25
            
            output_path = os.path.join(self.analysis_output_dir, filename)
            await self._run_in_executor(cv2.imwrite, output_path, summary_img)

        except Exception as e:
            logger.error(f"Không thể tạo panel '{filename}': {e}", exc_info=True)

    # =============================================================================

    
    async def _get_clothing_analysis_async(self, rgb_frame: np.ndarray, keypoints: np.ndarray) -> Optional[Dict]:
        """
        Pipeline phân tích quần áo: gọi module xử lý đầy đủ và trả về kết quả.
        """
        try:
            # Gọi hàm xử lý đầy đủ và trả về kết quả trực tiếp.
            # Việc phân loại (classification) đã được thực hiện BÊN TRONG hàm này rồi.
            analysis_result = await self.color_extractor.process_body_color_async(
                rgb_frame, 
                keypoints,
                self.clothing_classifier # Truyền classifier vào để nó sử dụng
            )
            return analysis_result
            
        except Exception as e: 
            logger.error(f"Lỗi trong pipeline phân tích trang phục: {e}", exc_info=True) 
            return None
    # ===================== KẾT THÚC PHẦN MÃ MỚI ĐƯỢC THÊM VÀO =====================

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
            
    # --------------------------------------------------------------------
    # HÀM XỬ LÝ CHÍNH (CORE PROCESSING)
    # --------------------------------------------------------------------

    # file: python/core/processing_RGBD.py

    async def process_human_async(self, frame_id: int, rgb_frame: np.ndarray, tof_depth_map: np.ndarray, box: tuple, keypoints: np.ndarray) -> Optional[Dict]:
        """
        [TỐI ƯU HOÀN CHỈNH] Luồng xử lý cho một người, tích hợp tất cả các bước phân tích.
        """
        async with self.semaphore:
            try:
                box_int = tuple(map(int, box[:4]))
                human_box_img = rgb_frame[box_int[1]:box_int[3], box_int[0]:box_int[2]]
                if human_box_img.size == 0: return None

                # --- BƯỚC 1: CHẠY SONG SONG CÁC PHÂN TÍCH ---
                gender_task = self._run_in_executor(self.gender_model.predict, human_box_img)
                world_data_task = self._run_in_executor(
                    self.stereo_projector.get_worldpoint_coordinates,
                    get_torso_box(keypoints, box_int), get_optimal_feet_point(keypoints, box_int), 
                    tof_depth_map, config.OPI_CONFIG.get("CAM_ANGLE_DEG")
                )
                
                gender_analysis, world_data = await asyncio.gather(gender_task, world_data_task, return_exceptions=True)

                # --- BƯỚC 2: XỬ LÝ KẾT QUẢ VÀ CÁC TÁC VỤ PHỤ THUỘC ---
                est_height_m, height_status = None, "N/A"
                if not isinstance(world_data, Exception) and world_data and world_data.get("distance_mm"):
                    est_height_m, height_status = await self._run_in_executor(
                        self.height_estimator.estimate, keypoints, world_data["distance_mm"] / 1000.0
                    )
                
                # Chạy phân tích quần áo SAU KHI có kết quả giới tính
                clothing_analysis = await self.color_extractor.process_and_classify(
                    image=rgb_frame, keypoints=keypoints, classifier=self.clothing_classifier, 
                    external_data={"gender_analysis": gender_analysis}
                )

                # --- BƯỚC 3: ĐÓNG GÓI TẤT CẢ THUỘC TÍNH ĐỂ GỬI CHO TRACKER ---
                person_attributes = {
                    "bbox": box, "map_keypoints": keypoints,
                    "gender_analysis": gender_analysis if not isinstance(gender_analysis, Exception) else {},
                    "clothing_analysis": clothing_analysis if not isinstance(clothing_analysis, Exception) else {},
                    "world_data": world_data if not isinstance(world_data, Exception) else {},
                    "est_height_m": est_height_m,
                    "height_status": height_status,
                }
                
                logger.info(f"✅ [Frame {frame_id}] Phân tích người hoàn tất: GT={(person_attributes['gender_analysis'] or {}).get('gender', 'N/A')}")
                
                # Tạo panel debug trong một tác vụ nền
                panel_name = f"panel_F{frame_id}_{datetime.now().strftime('%H%M%S%f')}.jpg"
                asyncio.create_task(self._create_analysis_panel_async(human_box_img.copy(), person_attributes, panel_name))

                return person_attributes

            except Exception as e:
                logger.error(f"Lỗi nghiêm trọng khi xử lý người cho frame {frame_id}: {e}", exc_info=True)
                return None
                
    async def _manage_people_count_state(self, current_person_count: int):
        """
        Quản lý bộ đệm đếm người và chỉ gửi đi giá trị ổn định nhất.
        1. Đưa số lượng người hiện tại vào một queue tạm.
        2. Khi queue tạm đủ số lượng (ví dụ: 5), lấy tất cả ra.
        3. Tìm giá trị xuất hiện nhiều nhất (mode) trong các giá trị đó.
        4. Gửi giá trị ổn định này vào queue chính để gửi qua WebSocket.
        """
        # 1. Đưa số lượng người hiện tại vào queue tạm
        await self.temp_count_queue.put(current_person_count)

        # 2. Kiểm tra nếu queue tạm đã đủ lớn để xử lý
        if self.temp_count_queue.qsize() >= self.count_buffer_size:
            counts_in_buffer = []
            # Lấy chính xác `count_buffer_size` phần tử từ queue
            for _ in range(self.count_buffer_size):
                try:
                    # Lấy phần tử ra khỏi queue mà không cần chờ
                    count = self.temp_count_queue.get_nowait()
                    counts_in_buffer.append(count)
                except asyncio.QueueEmpty:
                    # Trường hợp hiếm gặp nếu queue bị rỗng giữa chừng
                    break
            
            if not counts_in_buffer:
                return # Không có gì để xử lý

            # 3. Tìm giá trị xuất hiện nhiều nhất (mode)
            # Counter(counts_in_buffer).most_common(1) trả về list dạng [(giá_trị, số_lần_xuất_hiện)]
            # ví dụ: Counter([5, 5, 6, 5, 5]).most_common(1) -> [(5, 4)]
            # Chúng ta chỉ cần lấy giá trị (phần tử đầu tiên của tuple đầu tiên)
            stable_count = Counter(counts_in_buffer).most_common(1)[0][0]

            logger.info(f"Đã xử lý bộ đệm đếm: {counts_in_buffer} -> Gửi đi số lượng ổn định: {stable_count}")

            # 4. Gửi giá trị ổn định này vào queue chính
            packet = {"total_person": stable_count}
            asyncio.create_task(self.people_count_queue.put(packet))

    async def process_frame_queue(self, frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
        frame_number = 0
        while True:
            try:
                batch_data_paths = []
                try:
                    item = await asyncio.wait_for(frame_queue.get(), timeout=5.0)
                    if item is None: break
                    batch_data_paths.append(item); frame_queue.task_done()
                except asyncio.TimeoutError:
                    await self._manage_people_count_state(0)
                    continue

                while len(batch_data_paths) < self.batch_size:
                    try:
                        item = frame_queue.get_nowait()
                        if item is None: frame_queue.put_nowait(None); break
                        batch_data_paths.append(item); frame_queue.task_done()
                    except asyncio.QueueEmpty: break
                
                start_time = time.time()
                
                loaded_data_map = {}
                frames_for_detection = []
                for i, (rgb_path, depth_path, _) in enumerate(batch_data_paths):
                    try:
                        rgb_frame, depth_frame = cv2.imread(rgb_path), np.load(depth_path)
                        if rgb_frame is not None and depth_frame is not None:
                            frames_for_detection.append(rgb_frame)
                            loaded_data_map[i] = (rgb_frame, depth_frame)
                    except Exception as e: logger.error(f"Lỗi đọc file {rgb_path}: {e}")
                if not frames_for_detection: continue

                detection_results = await asyncio.gather(*[self._run_in_executor(self.detector.run_detection, f) for f in frames_for_detection], return_exceptions=True) 

                all_human_tasks = []
                people_count_per_frame = []

                for i, res in enumerate(detection_results):
                    if not res or isinstance(res, Exception):
                        people_count_per_frame.append(0)
                        continue
                    
                    kpts_data, boxes_data = res
                    rgb, depth = loaded_data_map[i]
                    
                    valid_detections_in_frame = 0
                    if kpts_data is not None and boxes_data is not None:
                        for kpts, box in zip(kpts_data, boxes_data):
                            if self._is_detection_valid(box, kpts, rgb.shape):
                                valid_detections_in_frame += 1
                                task = self.process_human_async(frame_number + i, rgb, depth, box, kpts)
                                all_human_tasks.append(task)
                    
                    people_count_per_frame.append(valid_detections_in_frame)

                current_person_count = people_count_per_frame[-1] if people_count_per_frame else 0
                await self._manage_people_count_state(current_person_count)

                if not all_human_tasks:
                    frame_number += len(frames_for_detection)
                    continue

                final_results = await asyncio.gather(*all_human_tasks)
                valid_final_results = [res for res in final_results if res] 
                
                if valid_final_results:
                    heights_cm = [res["est_height_m"] * 100.0 for res in valid_final_results if res.get("est_height_m")]
                    if heights_cm:
                        await self.height_queue.put({"table_id": self.table_id, "heights_cm": heights_cm})
                    
                    packet_for_tracker = {
                        "frame_id": frame_number, "time_detect": datetime.now().isoformat(),
                        "people_list": valid_final_results, "camera_id": device_id
                    }
                    await processing_queue.put(packet_for_tracker)
                    logger.info(f"[Frame {frame_number}] Đã gửi packet với {len(valid_final_results)} người tới tracker.")

                duration = time.time() - start_time
                if duration > 0:
                    current_fps = len(batch_data_paths) / duration
                    self.fps_avg = (self.fps_avg * self.call_count + current_fps) / (self.call_count + 1)
                    self.call_count += 1
                    logger.info(f"FPS Avg Processing: {self.fps_avg:.2f}")

                frame_number += len(frames_for_detection)
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong process_frame_queue: {e}", exc_info=True)

async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue, calib_path: str):
    logger.info("Khởi động worker xử lý...")
    try:
        processor = FrameProcessor(
            calib_path=calib_path,
            processing_queue=processing_queue,
            people_count_queue=people_count_queue,
            height_queue=height_queue
        )
        # [SỬA LỖI] Gọi hàm với đúng số lượng tham số
        await processor.process_frame_queue(frame_queue, processing_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True)