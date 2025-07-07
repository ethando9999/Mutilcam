# file: utils/yolo_pose_rknn.py (Phiên bản chuẩn hóa theo Rockchip và tối ưu)

import os
import numpy as np
import cv2
from rknn.api import RKNN
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# --- Các hằng số ---
CLASSES = ['person']
MODEL_PATH = "python/models/yolov8_pose.rknn" # <-- !!! ĐẢM BẢO ĐƯỜNG DẪN NÀY CHÍNH XÁC !!!
NMS_THRESH = 0.4
OBJECT_THRESH = 0.5

# --- Các hàm phụ trợ ---

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize và đệm ảnh để vừa với kích thước yêu cầu của model."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def dfl(x):
    """Giải mã Distribution Focal Loss (DFL)."""
    x = x.reshape(1, 4, 16).transpose(0, 2, 1)
    x = np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True) # Softmax
    x = np.sum(x * np.arange(16), axis=1)
    return x.reshape(-1)

class HumanDetection:
    """
    Lớp đóng gói cho việc phát hiện người và tư thế bằng YOLOv8-Pose trên RKNN.
    """
    def __init__(self, target='rk3588', core_mask=RKNN.NPU_CORE_AUTO):
        self.rknn = RKNN(verbose=False)
        self.load_model(MODEL_PATH)
        self.init_runtime(target, core_mask)

    def load_model(self, model_path):
        logger.info(f"Loading RKNN model from {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0: raise RuntimeError(f'Load RKNN model failed! ret={ret}')
        logger.info('Model loaded successfully.')

    def init_runtime(self, target, core_mask):
        logger.info('Initializing RKNN runtime...')
        ret = self.rknn.init_runtime(target=target, core_mask=core_mask)
        if ret != 0: raise RuntimeError(f'Init runtime failed! ret={ret}')
        logger.info('Runtime initialized successfully.')

    def postprocess(self, results, ratio, dw, dh):
        """
        Xử lý hậu kỳ đầu ra từ NPU để lấy bounding box và keypoints.
        Logic được chuẩn hóa theo mã nguồn gốc của Rockchip.
        """
        boxes, confs, keypoints = [], [], []
        
        # results[0-2] là các head cho bounding box, results[3] là cho keypoints
        box_outputs = [res.reshape(1, 65, -1) for res in results[:3]]
        kpts_output = results[3]

        for i in range(len(box_outputs)):
            stride = 8 * (2**i)
            reg_dist, conf_cls = np.split(box_outputs[i], [64], axis=1)
            
            # Lấy confidence của class 'person' và áp dụng sigmoid
            conf_cls = 1 / (1 + np.exp(-conf_cls[:, 0, :])) # Sigmoid
            
            mask = conf_cls > OBJECT_THRESH
            if not np.any(mask): continue
                
            # Lọc các detections hợp lệ
            reg_dist = reg_dist[mask] 
            conf_cls = conf_cls[mask]
            
            # Giải mã bounding box từ DFL
            xywh = np.apply_along_axis(dfl, 1, reg_dist.reshape(-1, 64)) * stride
            
            # Chuyển từ xywh (center, w, h) sang xyxy (top-left, bottom-right)
            xy, wh = xywh[:, :2], xywh[:, 2:]
            xyxy = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1)
            
            boxes.append(xyxy)
            confs.append(conf_cls)
            
            # Giải mã keypoints tương ứng
            kpts = kpts_output[mask].transpose(0, 2, 1) # (n, 17, 3)
            kpts[..., 0] *= stride
            kpts[..., 1] *= stride
            kpts[..., 2] = 1 / (1 + np.exp(-kpts[..., 2])) # Sigmoid
            keypoints.append(kpts)

        if not boxes: return [], []
            
        # NMS để loại bỏ các box trùng lặp
        all_boxes = np.concatenate(boxes, axis=0)
        all_confs = np.concatenate(confs, axis=0)
        all_keypoints = np.concatenate(keypoints, axis=0)
        
        indices = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_confs.tolist(), OBJECT_THRESH, NMS_THRESH)
        
        if len(indices) == 0: return [], []
        
        # Lấy các kết quả cuối cùng sau NMS
        final_boxes = all_boxes[indices]
        final_kpts = all_keypoints[indices]

        # Scale tọa độ về ảnh gốc (quan trọng nhất)
        final_boxes -= np.array([dw, dh, dw, dh])
        final_boxes /= ratio
        final_kpts[..., :2] -= np.array([dw, dh])
        final_kpts[..., :2] /= ratio
        
        return final_kpts, final_boxes

    def run_detection(self, img: np.ndarray) -> tuple[list, list]:
        """
        Hàm giao diện chính: nhận một ảnh, trả về keypoints và bounding boxes.
        """
        img_letterboxed, ratio, (dw, dh) = letterbox(img, (640, 640))
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        
        results = self.rknn.inference(inputs=[img_rgb])
        
        keypoints_data, boxes_data = self.postprocess(results, ratio, dw, dh)
        
        # Chuyển đổi sang định dạng list chuẩn để trả về
        return keypoints_data, [tuple(map(int, box)) for box in boxes_data]

    def release(self):
        self.rknn.release()