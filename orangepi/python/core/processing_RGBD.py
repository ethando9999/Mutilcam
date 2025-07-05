import asyncio
import cv2
from utils.logging_python_orangepi import get_logger
import time
from utils.yolo_pose_rknn import HumanDetection
from utils.pose_color_signature_new import PoseColorSignatureExtractor
from utils.cut_body_part import extract_body_parts_from_frame
from .stereo_projector import StereoProjector
from .keypoints_handle import get_head_center
import os

logger = get_logger(__name__)

class FrameProcessor:
    def __init__(self, batch_size=2): 
        self.detector = HumanDetection()
        self.pose_processor = PoseColorSignatureExtractor()
        self.stereo_projector = StereoProjector("config/calib_v2.npz")
        self.fps_avg = 0.0
        self.call_count = 0
        self.semaphore = asyncio.Semaphore(4) 
        self.batch_size = batch_size

    # (Các hàm async helper giữ nguyên)
    async def run_detection_async(self, frame):
        return await asyncio.to_thread(self.detector.run_detection, frame)
    
    async def process_body_parts_async(self, frame, keypoints):
        return await asyncio.to_thread(extract_body_parts_from_frame, frame, keypoints)

    async def get_depths_async(self, rgb_points, rgb_height, rgb_width, tof_depth_map):
        depths = await asyncio.to_thread(
            self.stereo_projector.get_depths_for_rgb_points,
            rgb_points, rgb_height, rgb_width, tof_depth_map
        )
        return depths

    # <<< HÀM ĐƯỢC CẬP NHẬT (Thêm điều kiện lọc theo độ sâu) >>>
    async def process_human_async(self, human_box, map_keypoints, frame_id, box, keypoints, tof_depth_map, rgb_height, rgb_width):
        """
        Xử lý một người, tính toán điểm đầu 3D và chỉ trả về kết quả nếu
        độ sâu của đầu < 4000mm (4 mét).
        """
        async with self.semaphore:
            try:
                head_center_kp = get_head_center(map_keypoints)
                head_point_3d = None

                # Chạy tác vụ trích xuất các bộ phận cơ thể song song
                body_parts_task = asyncio.create_task(self.process_body_parts_async(human_box, map_keypoints))

                if head_center_kp:
                    # Lấy độ sâu cho điểm trung tâm đầu
                    depths_task = asyncio.create_task(
                        self.get_depths_async([head_center_kp], rgb_height, rgb_width, tof_depth_map)
                    )
                    body_parts, depths_list = await asyncio.gather(body_parts_task, depths_task)
                    
                    if depths_list and depths_list[0] is not None:
                        head_depth = depths_list[0]

                        # =================================================================
                        # <<< THAY ĐỔI CHÍNH: LỌC THEO ĐỘ SÂU >>>
                        # Nếu độ sâu lớn hơn hoặc bằng 4000mm (4m), bỏ qua người này.
                        if head_depth >= 4000:
                            logger.debug(f"Bỏ qua người trong frame {frame_id} vì ở quá xa (độ sâu: {head_depth}mm).")
                            return None # Trả về None để không đưa vào hàng đợi xử lý
                        # =================================================================

                        # Nếu người ở trong phạm vi cho phép, tạo điểm 3D
                        head_point_3d = (head_center_kp[0], head_center_kp[1], head_depth)
                else:
                    # Nếu không có điểm đầu, vẫn lấy kết quả body_parts nhưng không có độ sâu
                    body_parts = await body_parts_task
                    logger.warning(f"Không tìm thấy trung tâm đầu cho người trong frame {frame_id}, bỏ qua tính toán độ sâu.")

                # Trả về cấu trúc dữ liệu hoàn chỉnh nếu người không bị lọc
                return {
                    "frame_id": frame_id,
                    "human_box": human_box,
                    "body_parts": body_parts,
                    "head_point_3d": head_point_3d,
                    "bbox": box,
                    "map_keypoints": map_keypoints,
                }
            
            except Exception as e:
                logger.error(f"Lỗi xử lý người cho frame {frame_id}: {e}", exc_info=True)
                return None

    # <<< HÀM NÀY KHÔNG CẦN THAY ĐỔI >>>
    # Logic `if result:` đã xử lý việc `process_human_async` trả về `None`
    async def process_frame_queue(self, frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
        """Xử lý khung hình từ hàng đợi, tối ưu hóa bằng cách chạy detection song song."""
        frame_number = 0
        while True:
            start_time = time.time()
            batch_frames = []
            try:
                item = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                cam_idx, frame_path, amp_path, depth_data = item
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        batch_frames.append((cam_idx, frame, depth_data))
                frame_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("process_frame_queue bị hủy.")
                break
            
            while len(batch_frames) < self.batch_size:
                try:
                    item = frame_queue.get_nowait()
                    cam_idx, frame_path, amp_path, depth_data = item
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            batch_frames.append((cam_idx, frame, depth_data))
                    frame_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            if not batch_frames:
                continue
            
            try:
                detection_tasks = [self.run_detection_async(frame) for _, frame, _ in batch_frames]
                detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)

                for i, (cam_idx, frame, depth_data) in enumerate(batch_frames):
                    detection_result = detection_results[i]
                    if isinstance(detection_result, Exception):
                        logger.error(f"Lỗi detection trên frame {i} của batch: {detection_result}")
                        continue
                    
                    keypoints_data, boxes_data = detection_result
                    
                    if not boxes_data:
                        logger.warning(f"Không phát hiện người trong khung hình từ camera {cam_idx}.")
                        continue

                    rgb_h, rgb_w, _ = frame.shape
                    human_processing_tasks = []
                    for index, (keypoints, box) in enumerate(zip(keypoints_data, boxes_data)):
                        human_box = frame[box[1]:box[3], box[0]:box[2]]
                        map_keypoints = self.detector.transform_keypoints_to_local(box, keypoints)
                        
                        task = asyncio.create_task(self.process_human_async(
                            human_box, map_keypoints, frame_number + i, box, keypoints, 
                            depth_data, rgb_h, rgb_w
                        ))
                        human_processing_tasks.append(task)

                    if human_processing_tasks:
                        results = await asyncio.gather(*human_processing_tasks, return_exceptions=True)
                        for result in results:
                            if isinstance(result, Exception):
                                logger.error(f"Lỗi trong xử lý người: {result}", exc_info=True)
                                continue
                            # Logic này không đổi: nếu result là None (do bị lọc), nó sẽ không được đưa vào hàng đợi.
                            if result:
                                result["camera_id"] = cam_idx
                                await processing_queue.put(result)

                frame_number += len(batch_frames)
                duration = time.time() - start_time
                fps_current = len(batch_frames) / duration if duration > 0 else 0
                self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
                self.call_count += 1
                if frame_number % 10 == 0:
                    logger.info(f"FPS xử lý: {self.fps_avg:.2f}, Hàng đợi vào: {frame_queue.qsize()}, Hàng đợi ra: {processing_queue.qsize()}")

            except asyncio.CancelledError:
                logger.info("Tác vụ xử lý batch bị hủy.")
                break
            except Exception as e:
                logger.error(f"Lỗi trong xử lý lô: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info("process_frame_queue kết thúc.")



async def start_processor(frame_queue: asyncio.Queue, processing_queue: asyncio.Queue):
    """Khởi động worker xử lý song song."""
    logger.info("Khởi động worker xử lý...")
    processor = FrameProcessor()
    task = asyncio.create_task(processor.process_frame_queue(frame_queue, processing_queue))
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Tác vụ xử lý đang được hủy...")
        task.cancel()
        await task
        logger.info("Tác vụ xử lý đã được hủy.")
    finally:
        thread_pool.shutdown(wait=True)