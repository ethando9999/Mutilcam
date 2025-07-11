import asyncio
import cv2
import os
from datetime import datetime
import numpy as np

# Giả định các module này tồn tại và hoạt động đúng
from utils.logging_python_orangepi import get_logger
from utils.yolo_pose import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector_final import StereoProjector
from .keypoints_handle import get_head_center, get_torso_box, adjust_keypoints_to_box
from .height_estimator import HeightEstimator
import config
import time 

logger = get_logger(__name__) 

class FrameProcessor:
    """
    Xử lý các lô frame RGB-D, từ phát hiện người cho đến trích xuất thông tin chi tiết.
    Phiên bản này được tối ưu để hoạt động ổn định và hiệu quả trên thiết bị biên.
    """
    def __init__(self, batch_size: int = 2):
        """
        Khởi tạo tất cả các module cần thiết cho việc xử lý.  
        """
        self.detector = HumanDetection()
        self.pose_processor = PoseColorSignatureExtractor() 
        self.stereo_projector = StereoProjector() 
        self.table_id = 1
        
        mtx_rgb = self.stereo_projector.params.get('mtx_rgb')
        if mtx_rgb is None:
            raise ValueError("mtx_rgb không có trong file hiệu chỉnh.")
        self.height_estimator = HeightEstimator(mtx_rgb)
        
        # Giới hạn số lượng người được xử lý song song để tránh quá tải
        self.semaphore = asyncio.Semaphore(4)
        self.batch_size = batch_size
        
        self.debug_dir = os.path.join(config.OPI_CONFIG.get("results_dir", "results"), "debug_projection")
        os.makedirs(self.debug_dir, exist_ok=True)
        logger.info("[DEBUG] Ảnh debug sẽ lưu vào: %s", self.debug_dir)
        logger.info("FrameProcessor (Tối ưu Toàn diện) đã được khởi tạo.")

    async def process_body_color_async(self, frame, keypoints):
        """Xử lý màu sắc cơ thể bất đồng bộ."""
        start_time = time.time()
        result = await self.pose_processor.process_body_color_async(frame, keypoints, True)
        logger.info(f"Xử lý màu sắc cơ thể mất {time.time() - start_time:.2f} giây")
        return result

    # --------------------------------------------------------------------
    # CÁC HÀM PHỤ TRỢ (HELPERS)
    # --------------------------------------------------------------------

    def _is_detection_valid(self, box: tuple, keypoints: np.ndarray, frame_shape: tuple) -> bool:
        """
        Xác thực chất lượng đầu vào từ YOLO.
        Một phát hiện chỉ hợp lệ khi cả Bounding Box và Keypoints đều đạt chuẩn.
        """
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        if np.any(np.isnan(box)) or np.any(np.isinf(box)):
            return False
        if x1 >= x2 or y1 >= y2:
            return False
        if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
            return False
        
        if keypoints.shape[1] >= 3:
            valid_kpts_count = np.sum(keypoints[:, 2] > 0.3)
        else:
            valid_kpts_count = np.sum(np.sum(keypoints[:, :2], axis=1) > 0)
        if valid_kpts_count < 4:
            logger.debug(f"Phát hiện có ít keypoints ({valid_kpts_count}). Bỏ qua.")
            return False
        return True

    async def _run_in_executor(self, func, *args):
        """Hàm helper để chạy các hàm blocking trong luồng riêng."""
        return await asyncio.to_thread(func, *args)

    async def save_debug_images(self, base_name, rgb_frame, tof_depth_frame, rgb_box, tof_box):
        """Lưu các ảnh gỡ lỗi một cách bất đồng bộ để không làm chậm luồng chính."""
        try:
            tof_color = cv2.normalize(tof_depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            tof_color_map = cv2.applyColorMap(tof_color, cv2.COLORMAP_JET)

            x1, y1, x2, y2 = rgb_box
            rgb_crop = rgb_frame[y1:y2, x1:x2]
            logger.warning(f"[DEBUG] rgb_box={rgb_box}, rgb_crop.shape={rgb_crop.shape}")
            if rgb_crop.size > 0:
                path = os.path.join(self.debug_dir, f"{base_name}_crop_rgb.png")
                success = await self._run_in_executor(cv2.imwrite, path, rgb_crop)
                logger.warning(f"[DEBUG] Ghi file '{path}' {'thành công' if success else 'thất bại'}")

            if tof_box:
                tx1, ty1, tx2, ty2 = tof_box
                tof_crop = tof_color_map[ty1:ty2, tx1:tx2]
                logger.warning(f"[DEBUG] tof_box={tof_box}, tof_crop.shape={tof_crop.shape}")
                if tof_crop.size > 0:
                    path = os.path.join(self.debug_dir, f"{base_name}_crop_tof.png")
                    success = await self._run_in_executor(cv2.imwrite, path, tof_crop)
                    logger.warning(f"[DEBUG] Ghi file '{path}' {'thành công' if success else 'thất bại'}")

                cv2.rectangle(tof_color_map, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            path_rgb_full = os.path.join(self.debug_dir, f"{base_name}_full_rgb_boxed.png")
            success_rgb = await self._run_in_executor(cv2.imwrite, path_rgb_full, rgb_frame)
            logger.warning(f"[DEBUG] Ghi file '{path_rgb_full}' {'thành công' if success_rgb else 'thất bại'}")

            path_tof_full = os.path.join(self.debug_dir, f"{base_name}_full_tof_boxed.png")
            success_tof = await self._run_in_executor(cv2.imwrite, path_tof_full, tof_color_map)
            logger.warning(f"[DEBUG] Ghi file '{path_tof_full}' {'thành công' if success_tof else 'thất bại'}")

        except Exception:
            logger.exception("Lỗi khi lưu ảnh debug")

    # --------------------------------------------------------------------
    # HÀM XỬ LÝ CHÍNH (CORE PROCESSING)
    # --------------------------------------------------------------------

    async def process_human_async(self, frame_id: int, rgb_frame: np.ndarray, tof_depth_map: np.ndarray, box: tuple, keypoints: np.ndarray):
        """
        Luồng xử lý tối ưu cho một người được phát hiện.
        """
        async with self.semaphore:
            try:
                torso_box = get_torso_box(keypoints, box)
                distance_mm, status = await self._run_in_executor(self.stereo_projector.get_robust_distance, torso_box, tof_depth_map)

                if status != "OK" or distance_mm is None or not (500 < distance_mm < 4000):
                    return None

                logger.info(f"✅ Người hợp lệ trong phạm vi 4m. Khoảng cách: {distance_mm/1000:.2f}m. Bắt đầu xử lý sâu...")
                distance_m = distance_mm / 1000.0

                height_task = self._run_in_executor(self.height_estimator.estimate, keypoints, distance_m)

                human_box_img = rgb_frame[box[1]:box[3], box[0]:box[2]]
                if human_box_img.size == 0:
                    return None

                adjusted_keypoints = adjust_keypoints_to_box(keypoints, box)
                body_parts_task = self._run_in_executor(extract_body_parts_from_frame, human_box_img, adjusted_keypoints)
                projection_task = self._run_in_executor(self.stereo_projector.project_rgb_box_to_tof, box, tof_depth_map)

                (est_height_m, height_status), body_parts_boxes, tof_box_projected = await asyncio.gather(
                    height_task, body_parts_task, projection_task
                )

                logger.warning(f"[DEBUG] tof_box_projected = {tof_box_projected!r}")
                if tof_box_projected:
                    debug_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}_F{frame_id}"
                    # Đợi lưu ảnh debug hoàn thành
                    await self.save_debug_images(debug_base_name, rgb_frame.copy(), tof_depth_map.copy(), box, tof_box_projected)

                head_center_kp = get_head_center(keypoints)
                world_point_3d = None

                if head_center_kp is not None and distance_mm is not None:
                    pt = np.array([[head_center_kp]], dtype=np.float32)
                    undistorted = cv2.undistortPoints(
                        pt,
                        self.stereo_projector.mtx_rgb,
                        self.stereo_projector.dist_rgb,
                        None,
                        self.stereo_projector.mtx_rgb
                    )
                    head_points_3d = self.stereo_projector._back_project_to_3d(undistorted, distance_mm)
                    if head_points_3d is not None and head_points_3d.shape[0] > 0:
                        world_point_3d = head_points_3d[0]

                distance_str = f"{distance_mm:.2f}mm" if distance_mm is not None else "N/A"
                height_str = f"{est_height_m:.2f}m" if est_height_m is not None else "N/A"

                logger.info(f"✅✅ Xử lý hoàn tất: Khoảng cách={distance_str}, Chiều cao={height_str} ({height_status}), head_point_3d: {world_point_3d}")
                if world_point_3d is not None:
                    logger.info(f"Tọa độ đầu 3D (World Space, mét): {world_point_3d}")

                time_detect = datetime.now().isoformat()
                return {
                    "frame_id": frame_id,
                    "human_box": human_box_img,
                    "body_parts": body_parts_boxes,
                    "body_color": None,
                    "head_point_3d": world_point_3d,
                    "bbox": box,
                    "map_keypoints": keypoints,
                    "distance_mm": distance_mm,
                    "est_height_m": est_height_m,
                    "height_status": height_status,
                    "time_detect": time_detect,
                }

            except Exception as e:
                logger.error(f"Lỗi xử lý người cho frame {frame_id}: {e}", exc_info=True)
                return None

    async def process_frame_queue(self, frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue):
        frame_number = 0
        while True:
            batch_data_paths = []
            try:
                item = await asyncio.wait_for(frame_queue.get(), timeout=5.0)
                if item is None: break
                batch_data_paths.append(item); frame_queue.task_done()
            except (asyncio.TimeoutError, asyncio.CancelledError): continue
            
            while len(batch_data_paths) < self.batch_size:
                try:
                    item = frame_queue.get_nowait()
                    if item is None: frame_queue.put_nowait(None); break 
                    batch_data_paths.append(item); frame_queue.task_done()
                except asyncio.QueueEmpty: break
            if not batch_data_paths: continue

            try:
                loaded_data_map = {}
                frames_for_detection = []
                for i, (rgb_path, depth_path, amp_path) in enumerate(batch_data_paths):
                    try:
                        rgb_frame = cv2.imread(rgb_path)
                        depth_frame = np.load(depth_path) if depth_path and os.path.exists(depth_path) else None
                        if rgb_frame is not None and depth_frame is not None:
                            frames_for_detection.append(rgb_frame)
                            loaded_data_map[i] = (rgb_frame, depth_frame)
                    except Exception as e:
                        logger.error(f"Lỗi khi đọc file dữ liệu {rgb_path}: {e}")
                if not frames_for_detection: continue

                detection_results = await asyncio.gather(*[self._run_in_executor(self.detector.run_detection, f) for f in frames_for_detection], return_exceptions=True)

                max_people_counting = 0
                all_human_tasks = []
                for i, detection_result in enumerate(detection_results):
                    if not detection_result or isinstance(detection_result, Exception): continue
                    keypoints_data, boxes_data = detection_result
                    num_detected = len(keypoints_data)
                    if num_detected > max_people_counting:
                        max_people_counting = num_detected
                    original_rgb, original_depth = loaded_data_map[i]
                    for kpts, box in zip(keypoints_data, boxes_data):
                        if not self._is_detection_valid(box, kpts, original_rgb.shape):
                            logger.warning(f"Phát hiện không hợp lệ (box hoặc keypoints). Box: {box}. Bỏ qua.")
                            continue
                        all_human_tasks.append(
                            self.process_human_async(frame_number + i, original_rgb, original_depth, box, kpts)
                        )

                await people_count_queue.put({"total_person": max_people_counting})
                if all_human_tasks:
                    processing_results = await asyncio.gather(*all_human_tasks, return_exceptions=True)
                    all_valid_heights_cm = []
                    for result in processing_results:
                        if result and not isinstance(result, Exception):
                            result["camera_id"] = config.OPI_CONFIG.get("device_id", "opi_01")
                            height_m = result.get("est_height_m")
                            if height_m is not None:
                                height_cm = height_m * 100.0
                                if 140.0 <= height_cm <= 190.0:
                                    all_valid_heights_cm.append(height_cm)
                            await processing_queue.put(result)

                    await height_queue.put({"table_id": self.table_id, "heights_cm": all_valid_heights_cm})
                frame_number += len(frames_for_detection)
            except Exception as e:
                logger.error(f"Lỗi không mong muốn trong xử lý lô: {e}", exc_info=True)

async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue):
    logger.info("Khởi động worker xử lý (Phiên bản cuối cùng)...")
    try:
        processor = FrameProcessor(batch_size=2)
        await processor.process_frame_queue(frame_queue, processing_queue, people_count_queue, height_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True) 
