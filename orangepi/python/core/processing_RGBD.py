# file: python/core/processing_RGBD.py

import asyncio
import cv2
import os
from datetime import datetime
import numpy as np
import time
from typing import Optional
from collections import Counter 

# Giả định các module này tồn tại và hoạt động đúng
from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector import StereoProjector
from .keypoints_handle import get_head_center, get_torso_box, adjust_keypoints_to_box
from .height_estimator import HeightEstimator
import config

logger = get_logger(__name__)

class FrameProcessor:
    """
    Xử lý các lô frame RGB-D, từ phát hiện người cho đến trích xuất thông tin chi tiết.
    Tích hợp logic gửi dữ liệu qua WebSocket một cách thông minh và làm mịn dữ liệu.
    """
    def __init__(self, calib_path: str, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue, batch_size: int = 5):
        """
        Khởi tạo tất cả các module cần thiết và các biến trạng thái.
        """
        # ... (các phần khởi tạo detector, pose_processor, stereo_projector, height_estimator không đổi)
        self.detector = HumanDetection()
        self.pose_processor = PoseColorSignatureExtractor()
        self.stereo_projector = StereoProjector(calib_file_path=calib_path)
        
        mtx_rgb = self.stereo_projector.params.get('mtx_rgb')
        if mtx_rgb is None: raise ValueError("mtx_rgb không có trong file hiệu chỉnh.")
        self.height_estimator = HeightEstimator(mtx_rgb)
        
        # Lưu lại các queue để sử dụng
        self.people_count_queue = people_count_queue
        self.height_queue = height_queue
        
        self.semaphore = asyncio.Semaphore(4)
        self.batch_size = batch_size
        
        self.debug_dir = os.path.join(config.OPI_CONFIG.get("results_dir", "results"), "debug_projection")
        os.makedirs(self.debug_dir, exist_ok=True)

        # --- Quản lý bộ đệm đếm người (THAY THẾ CHO LOGIC CŨ) ---
        self.temp_count_queue = asyncio.Queue()
        self.count_buffer_size = 5 # Kích thước bộ đệm như yêu cầu

        # Các biến cũ không còn cần thiết
        # self.is_in_no_people_state = True
        # self.last_zero_sent_time = 0
        # self.PERIODIC_ZERO_INTERVAL = 10 

        self.pass_count = -1
        self.fps_avg = 0.0
        self.call_count = 0    
        self.table_id = config.OPI_CONFIG.get("SOCKET_TABLE_ID", 1)
        
        logger.info("FrameProcessor (Tối ưu & Tích hợp Websocket với bộ đệm) đã được khởi tạo.")
    # --------------------------------------------------------------------
    # CÁC HÀM PHỤ TRỢ (HELPERS)
    # --------------------------------------------------------------------

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

    # --------------------------------------------------------------------
    # HÀM XỬ LÝ CHÍNH (CORE PROCESSING)
    # --------------------------------------------------------------------

    async def process_human_async(self, frame_id: int, rgb_frame: np.ndarray, tof_depth_map: np.ndarray, box: tuple, keypoints: np.ndarray, precomputed_distance_mm: Optional[float] = None):
        """
        Luồng xử lý tối ưu cho một người. Tái sử dụng khoảng cách đã tính để tăng hiệu suất.
        """
        async with self.semaphore:
            try:
                distance_mm = precomputed_distance_mm
                if distance_mm is None: # Dự phòng nếu không có khoảng cách tính sẵn
                    torso_box = get_torso_box(keypoints, box)
                    distance_mm, status = await self._run_in_executor(self.stereo_projector.get_robust_distance, torso_box, tof_depth_map)
                    if status != "OK" or not (100 < distance_mm < 4000):
                        return None
                
                logger.info(f"✅ Người hợp lệ. Khoảng cách: {distance_mm/1000:.2f}m. Bắt đầu xử lý sâu...")
                distance_m = distance_mm / 1000.0

                # Chạy song song các tác vụ còn lại
                height_task = self._run_in_executor(self.height_estimator.estimate, keypoints, distance_m)
                
                human_box_img = rgb_frame[box[1]:box[3], box[0]:box[2]]
                if human_box_img.size == 0: return None
                
                adjusted_keypoints = adjust_keypoints_to_box(keypoints, box)
                body_parts_task = self._run_in_executor(extract_body_parts_from_frame, human_box_img, adjusted_keypoints)
                projection_task = self._run_in_executor(self.stereo_projector.project_rgb_box_to_tof, box, distance_mm, tof_depth_map.shape)

                (est_height_m, height_status), body_parts_boxes, tof_box_projected = await asyncio.gather(
                    height_task, body_parts_task, projection_task
                )

                if tof_box_projected:
                    debug_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}_F{frame_id}"
                    asyncio.create_task(self.save_debug_images(debug_base_name, rgb_frame.copy(), tof_depth_map.copy(), box, tof_box_projected))

                head_kp = get_head_center(keypoints)
                world_point_3d = None
                if head_kp is not None:
                    head_kp = np.array(head_kp, dtype=np.float32).reshape(1, 2)
                    head_kp_3d = self.stereo_projector._back_project_rgb_to_3d(head_kp, distance_mm)
                    world_point_3d = head_kp_3d[0]
                
                logger.info(f"✅✅ Xử lý hoàn tất: Khoảng cách={distance_m:.2f}m, "
                            f"Chiều cao={(f'{est_height_m:.2f}' if est_height_m is not None else 'None')}m ({height_status})")

                return {
                    "frame_id": frame_id, 
                    "human_box": human_box_img, 
                    "body_parts": body_parts_boxes,
                    "head_point_3d": world_point_3d, 
                    "bbox": box, "map_keypoints": keypoints,
                    "distance_mm": distance_mm, 
                    "est_height_m": est_height_m, 
                    "height_status": height_status,
                    "time_detect": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Lỗi xử lý người cho frame {frame_id}: {e}", exc_info=True)
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
        """
        Vòng lặp chính: Lấy dữ liệu, điều phối tác vụ và gửi kết quả một cách tối ưu.
        """
        frame_number = 0
        while True:
            start_time = time.time()
            # BƯỚC 1: LẤY VÀ ĐỌC DỮ LIỆU BATCH FRAME
            batch_data_paths = []
            try:
                item = await asyncio.wait_for(frame_queue.get(), timeout=5.0)
                if item is None: break
                batch_data_paths.append(item); frame_queue.task_done()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Nếu không có frame nào, vẫn chạy logic gửi số 0 định kỳ
                await self._manage_people_count_state(0)
                continue

            while len(batch_data_paths) < self.batch_size:
                try:
                    item = frame_queue.get_nowait()
                    if item is None: frame_queue.put_nowait(None); break
                    batch_data_paths.append(item); frame_queue.task_done()
                except asyncio.QueueEmpty: break
            
            try:
                loaded_data_map = {}
                frames_for_detection = []
                for i, (rgb_path, depth_path, amp_path) in enumerate(batch_data_paths):
                    # ... (logic đọc file)
                    try:
                        rgb_frame, depth_frame = cv2.imread(rgb_path), np.load(depth_path)
                        if rgb_frame is not None and depth_frame is not None:
                            frames_for_detection.append(rgb_frame)
                            loaded_data_map[i] = (rgb_frame, depth_frame)
                    except Exception as e: logger.error(f"Lỗi đọc file {rgb_path}: {e}")
                if not frames_for_detection: continue

                # BƯỚC 2: PHÁT HIỆN NGƯỜI (SONG SONG)
                detection_results = await asyncio.gather(*[self._run_in_executor(self.detector.run_detection, f) for f in frames_for_detection], return_exceptions=True) 

                # BƯỚC 3: LỌC VÀ TÍNH KHOẢNG CÁCH (SONG SONG) 
                potential_detections = []
                for i, res in enumerate(detection_results):
                    if not res or isinstance(res, Exception): continue
                    kpts_data, boxes_data = res
                    rgb, depth = loaded_data_map[i]
                    for kpts, box in zip(kpts_data, boxes_data):
                        if self._is_detection_valid(box, kpts, rgb.shape):
                            potential_detections.append({"idx": i, "rgb": rgb, "depth": depth, "box": box, "kpts": kpts})

                distance_tasks = [self._run_in_executor(self.stereo_projector.get_robust_distance, get_torso_box(d["kpts"], d["box"]), d["depth"]) for d in potential_detections]
                distance_results = await asyncio.gather(*distance_tasks, return_exceptions=True)

                # BƯỚC 4: TẠO TÁC VỤ XỬ LÝ SÂU
                all_human_tasks = []
                valid_counts_per_frame = [0] * len(frames_for_detection)
                for i, det_info in enumerate(potential_detections):
                    dist_res = distance_results[i]
                    if isinstance(dist_res, Exception): continue
                    dist_mm, status = dist_res
                    if status == "OK" and dist_mm and (100 < dist_mm < 4000):
                        valid_counts_per_frame[det_info["idx"]] += 1
                        all_human_tasks.append(self.process_human_async(frame_number + det_info["idx"], det_info["rgb"], det_info["depth"], det_info["box"], det_info["kpts"], precomputed_distance_mm=dist_mm))

                # BƯỚC 5: GỬI KẾT QUẢ TỔNG HỢP VÀ CHI TIẾT
                max_person_count = max(valid_counts_per_frame) if valid_counts_per_frame else 0
                await self._manage_people_count_state(max_person_count)

                if all_human_tasks:
                    final_results = await asyncio.gather(*all_human_tasks, return_exceptions=True)
                    valid_final_results = [res for res in final_results if res and not isinstance(res, Exception)]
                    
                    if valid_final_results:
                        heights_cm = []
                        for res in valid_final_results:
                            h = res.get("est_height_m")
                            if h is not None:
                                heights_cm.append(h * 100.0)
                            else:
                                logger.warning(f"Bỏ qua kết quả vì est_height_m bị None")

                        height_packet = {"table_id": self.table_id, "heights_cm": heights_cm}
                        asyncio.create_task(self.height_queue.put(height_packet))
                        
                        for result in valid_final_results:
                            result["camera_id"] = config.OPI_CONFIG.get("device_id", "opi_01")
                            await processing_queue.put(result)
                            logger.info("Đã put data vào processing_queue")
                
                frame_number += len(frames_for_detection)
                end_time = time.time()
                duration = end_time - start_time
                fps_current = len(batch_data_paths) / duration if duration > 0 else 0
                # Giả sử self.fps_avg, self.call_count đã khởi tạo trước đó
                self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
                self.call_count += 1
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong xử lý lô: {e}", exc_info=True)

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
            batch_size=4 
        )
        await processor.process_frame_queue(frame_queue, processing_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True) 