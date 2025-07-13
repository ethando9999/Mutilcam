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
from .stereo_projector import StereoProjector
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
    def __init__(self, batch_size: int = 1): 
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
        # luôn gửi 0 và 1
        if count in (0, 1):
            await queue.put({"total_person": count})
            self._prev_person_count = count
            self._repeat_count = 0
            return

        # khởi tạo lần đầu
        if self._prev_person_count is None:
            self._prev_person_count = count
            self._repeat_count = 1
            await queue.put({"total_person": count})
            return

        # nếu giá trị thay đổi, reset counter và gửi lần đầu
        if self._prev_person_count != count:
            self._prev_person_count = count
            self._repeat_count = 1
            await queue.put({"total_person": count})
            return

        # nếu lặp lại, tăng đếm và chỉ gửi khi đếm ≥ 2
        self._repeat_count += 1
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

    async def save_debug_images(self, base_name, rgb_frame, tof_depth_frame, rgb_box, tof_box:None):
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
        box: tuple[int,int,int,int],
        keypoints: np.ndarray,
        *,
        precomputed_distance_mm: float | None = None
    ) -> dict | None:
        """
        Xử lý thông tin người: dùng precomputed distance tái sử dụng, kiểm tra FOV ToF, ước lượng chiều cao, cắt body parts.
        """
        async with self.semaphore:
            try:
                # B1: Lấy khoảng cách
                if precomputed_distance_mm is None:
                    torso = get_torso_box(keypoints, box)
                    distance_mm, status = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.stereo_projector.get_robust_distance,
                        torso,
                        tof_depth_map
                    )
                    if status != "OK" or distance_mm is None or not (0 < distance_mm < 4000):
                        logger.info(f"Có người ngoài 4m, distance_mm {distance_mm} status: {status}")
                        debug_name = f"{datetime.now():%Y%m%d_%H%M%S%f}_F{frame_id}"
                        await self.save_debug_images(debug_name, rgb_frame, tof_depth_map, box, None)
                        return None
                else:
                    distance_mm = precomputed_distance_mm

                # B2: Kiểm tra FOV ToF sử dụng precomputed distance
                tof_box = self.stereo_projector.project_rgb_box_to_tof(box, tof_depth_map, distance_mm)
                if tof_box is None:
                    logger.info(f"Frame {frame_id}: hộp ngoài FOV ToF.")
                    return None

                # B3: Cắt ROI RGB
                x1, y1, x2, y2 = box
                human_roi = rgb_frame[y1:y2, x1:x2]
                if human_roi.size == 0:
                    return None

                # B4: Chạy bất đồng bộ các tác vụ nặng
                loop = asyncio.get_event_loop()
                h_task = loop.run_in_executor(None, self.height_estimator.estimate, keypoints, distance_mm/1000.0)
                body_parts_task = loop.run_in_executor(None, extract_body_parts_from_frame, human_roi, adjust_keypoints_to_box(keypoints, box))

                # B5: Nhận kết quả
                est_height_m, height_status = await h_task
                body_parts_boxes = await body_parts_task

                # B6: Tính head 3D
                head_kp = get_head_center(keypoints)
                world_point_3d = None
                if head_kp is not None:
                    pt = np.array([[head_kp]], dtype=np.float32)
                    und = cv2.undistortPoints(pt, self.stereo_projector.mtx_rgb, self.stereo_projector.dist_rgb, None, self.stereo_projector.mtx_rgb)
                    hp3d = self.stereo_projector._back_project(und.reshape(-1,2), distance_mm)
                    if hp3d.size:
                        world_point_3d = hp3d[0]

                # B7: Debug images
                debug_name = f"{datetime.now():%Y%m%d_%H%M%S%f}_F{frame_id}"
                await self.save_debug_images(debug_name, rgb_frame, tof_depth_map, box, tof_box)

                # B8: Trả kết quả tối ưu
                return {
                    "frame_id": frame_id,
                    "bbox": box,
                    "distance_mm": distance_mm,
                    "est_height_m": est_height_m,
                    "height_status": height_status,
                    "body_parts": body_parts_boxes,
                    "tof_box": tof_box,
                    "head_point_3d": world_point_3d,
                    "time_detect": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error(f"Error processing human frame {frame_id}: {e}", exc_info=True)
                return None
            
    async def process_frame_queue(self,
                                    frame_queue: asyncio.Queue,
                                    processing_queue: asyncio.Queue,
                                    people_count_queue: asyncio.Queue,
                                    height_queue: asyncio.Queue):
        frame_number = 0
        while True:
            # 1. Lấy 1 frame từ queue
            try:
                item = await frame_queue.get()
                if item is None:
                    break
                rgb_path, depth_path, _ = item
                frame_queue.task_done()
            except Exception:
                continue

            # 2. Đọc dữ liệu
            try:
                rgb = cv2.imread(rgb_path)
                depth = np.load(depth_path) if depth_path and os.path.exists(depth_path) else None
                if rgb is None or depth is None:
                    continue
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                continue

            # 3. Phát hiện người
            try:
                det = await self._run_in_executor(self.detector.run_detection, rgb)
            except Exception as e:
                logger.error(f"Detection error: {e}")
                det = None

            # 4. Xác thực và tạo task xử lý
            proc_tasks = []
            if det:
                keypoints, boxes = det
                logger.warning(f"[DEBUG] yolo phat hien tren anh rgb: {len(keypoints)} nguoi")
                for kpts, box in zip(keypoints, boxes):
                    if not self._is_detection_valid(box, kpts, rgb.shape):
                        continue
                    torso = get_torso_box(kpts, box)
                    proc_tasks.append(
                        self.process_human_async(
                            frame_number,
                            rgb,
                            depth,
                            torso,
                            kpts
                        )
                    )
                

            # 5. Thực thi xử lý con người và tính số người hợp lệ
            heights_cm = []
            valid_results = []
            if proc_tasks:
                results = await asyncio.gather(*proc_tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, dict):
                        valid_results.append(res)
                        h_m = res.get("est_height_m")
                        if h_m:
                            heights_cm.append(h_m * 100)
                        res["camera_id"] = config.OPI_CONFIG.get("device_id", "opi_01")
                        await processing_queue.put(res)

            # 6. Cập nhật số người dựa trên valid_results
            count = len(valid_results)
            await self._check_and_put_count(count, people_count_queue)

            # 7. Gửi chiều cao
            await height_queue.put({"table_id": self.table_id, "heights_cm": heights_cm})

            frame_number += 1

async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue, people_count_queue: asyncio.Queue, height_queue: asyncio.Queue):
    logger.info("Khởi động worker xử lý (Phiên bản cuối cùng)...")
    try:
        processor = FrameProcessor(batch_size=2)
        await processor.process_frame_queue(frame_queue, processing_queue, people_count_queue, height_queue)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng trong start_processor: {e}", exc_info=True) 
