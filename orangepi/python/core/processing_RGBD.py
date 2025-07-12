import asyncio
import cv2
import os
from datetime import datetime
import numpy as np
from typing import Optional

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
        
        # Biến để kiểm soát chỉ gửi khi max_person_count lặp lại lần thứ 2 liên tiếp
        self._prev_person_count = None
        self._repeat_count = 0
        
        self.debug_dir = os.path.join(config.OPI_CONFIG.get("results_dir", "results_tof"))
        os.makedirs(self.debug_dir, exist_ok=True)
        logger.info("[DEBUG] Ảnh debug sẽ lưu vào: %s", self.debug_dir)
        logger.info("FrameProcessor (Tối ưu Toàn diện) đã được khởi tạo.")

    async def _check_and_put_count(self, count: int, queue: asyncio.Queue):
        """
        Chỉ put count vào queue khi giá trị count lặp lại lần thứ 2 liên tiếp,
        trừ khi count là 0 hoặc 1 thì luôn gửi.
        """
        if count in [0, 1]:
            await queue.put({"total_person": count})
            self._prev_person_count = count
            self._repeat_count = 0
            return

        if self._prev_person_count == count:
            self._repeat_count += 1
        else:
            self._prev_person_count = count
            self._repeat_count = 1

        if self._repeat_count >= 2:
            await queue.put({"total_person": count})
            
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
        """Lưu ảnh RGB, TOF màu full và file depth raw vào thư mục riêng biệt."""
        try:
            # Chuyển depth thành ảnh màu để dễ debug
            tof_color = cv2.normalize(tof_depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            tof_color_map = cv2.applyColorMap(tof_color, cv2.COLORMAP_JET)

            # Vẽ box lên ảnh RGB
            x1, y1, x2, y2 = rgb_box
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ box lên ảnh TOF nếu có
            if tof_box:
                tx1, ty1, tx2, ty2 = tof_box
                cv2.rectangle(tof_color_map, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

            os.makedirs(self.debug_dir, exist_ok=True)
            # Đường dẫn thư mục con
            rgb_dir = os.path.join(self.debug_dir, "rgb_full")
            tof_dir = os.path.join(self.debug_dir, "tof_full")
            raw_dir = os.path.join(self.debug_dir, "tof_raw")

            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(tof_dir, exist_ok=True)
            os.makedirs(raw_dir, exist_ok=True)

            # Lưu ảnh RGB có box
            path_rgb_full = os.path.join(rgb_dir, f"{base_name}_rgb_boxed.png")
            success_rgb = await self._run_in_executor(cv2.imwrite, path_rgb_full, rgb_frame)
            logger.warning(f"[DEBUG] Ghi file '{path_rgb_full}' {'thành công' if success_rgb else 'thất bại'}")

            # Lưu ảnh TOF màu có box
            path_tof_full = os.path.join(tof_dir, f"{base_name}_tof_boxed.png")
            success_tof = await self._run_in_executor(cv2.imwrite, path_tof_full, tof_color_map)
            logger.warning(f"[DEBUG] Ghi file '{path_tof_full}' {'thành công' if success_tof else 'thất bại'}")

            # Lưu depth raw
            path_npz = os.path.join(raw_dir, f"{base_name}_tof_depth.npz")
            np.savez_compressed(path_npz, tof_depth=tof_depth_frame)
            logger.warning(f"[DEBUG] Đã lưu file depth .npz tại '{path_npz}'")

        except Exception:
            logger.exception("Lỗi khi lưu ảnh debug")

    # --------------------------------------------------------------------
    # HÀM XỬ LÝ CHÍNH (CORE PROCESSING)
    # --------------------------------------------------------------------

    async def process_human_async(
        self,
        frame_id: int,
        rgb_frame: np.ndarray,
        tof_depth_map: np.ndarray,
        box: tuple,
        keypoints: np.ndarray,
        *,
        precomputed_distance_mm: Optional[float] = None
    ):
        """
        Xử lý thông tin người đã được phát hiện. Nếu đã tính sẵn distance_mm thì không cần tính lại.
        """
        async with self.semaphore:
            try:
                # --- Dùng distance_mm được truyền vào hoặc tính lại nếu không có ---
                if precomputed_distance_mm is not None:
                    distance_mm = precomputed_distance_mm
                else:
                    torso_box = get_torso_box(keypoints, box)
                    distance_mm, status = await self._run_in_executor(
                        self.stereo_projector.get_robust_distance,
                        torso_box,
                        tof_depth_map
                    )
                    if status != "OK" or distance_mm is None or not (500 < distance_mm < 3500):
                        return None

                logger.info(f"✅ Người hợp lệ. Khoảng cách: {distance_mm/1000:.2f}m. Bắt đầu xử lý sâu...")

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

                if tof_box_projected:
                    debug_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}_F{frame_id}"
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

                logger.info(f"✅✅ Xử lý hoàn tất: Khoảng cách={distance_mm:.2f}mm, Chiều cao={est_height_m:.2f}m ({height_status}), head_point_3d={world_point_3d}") 

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
                    "time_detect": datetime.now().isoformat(),
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

            if not batch_data_paths:
                continue

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
                if not frames_for_detection:
                    continue

                detection_results = await asyncio.gather(*[
                    self._run_in_executor(self.detector.run_detection, f)
                    for f in frames_for_detection
                ], return_exceptions=True)

                per_frame_counts = []
                all_human_tasks = []
                all_valid_heights_cm = []

                for i, detection_result in enumerate(detection_results):
                    if not detection_result or isinstance(detection_result, Exception):
                        per_frame_counts.append(0)
                        continue

                    keypoints_data, boxes_data = detection_result
                    original_rgb, original_depth = loaded_data_map[i]

                    distance_tasks = []
                    valid_detections_info = []
                    for kpts, box in zip(keypoints_data, boxes_data):
                        if not self._is_detection_valid(box, kpts, original_rgb.shape):
                            continue
                        torso_box = get_torso_box(kpts, box)
                        distance_tasks.append(
                            self._run_in_executor(self.stereo_projector.get_robust_distance, torso_box, original_depth)
                        )
                        valid_detections_info.append((box, kpts))

                    distance_results = await asyncio.gather(*distance_tasks, return_exceptions=True)

                    valid_count_this_frame = 0
                    for idx, res in enumerate(distance_results):
                        if isinstance(res, Exception):
                            continue
                        distance_mm, status = res
                        if status != "OK" or distance_mm is None or not (500 < distance_mm < 3500):
                            continue

                        valid_count_this_frame += 1
                        box, kpts = valid_detections_info[idx] 
                        all_human_tasks.append(
                            self.process_human_async(
                                frame_number + i,
                                original_rgb,
                                original_depth,
                                box,
                                kpts,
                                precomputed_distance_mm=distance_mm
                            )
                        )
                    per_frame_counts.append(valid_count_this_frame)

                logger.info("per_frame_counts: %s", per_frame_counts)
                max_person_count = max(per_frame_counts) if per_frame_counts else 0
                # Sử dụng hàm kiểm tra mới
                await self._check_and_put_count(max_person_count, people_count_queue)

                if all_human_tasks:
                    processing_results = await asyncio.gather(*all_human_tasks, return_exceptions=True)
                    for result in processing_results:
                        if result and not isinstance(result, Exception):
                            result["camera_id"] = config.OPI_CONFIG.get("device_id", "opi_01")
                            height_m = result.get("est_height_m")
                            if height_m is not None:
                                height_cm = height_m * 100.0
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
