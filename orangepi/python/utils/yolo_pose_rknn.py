import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2
import math
from rknn.api import RKNN

from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Các hằng số toàn cục
CLASSES = ['person']

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)

kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

core_mask = [RKNN.NPU_CORE_AUTO, RKNN.NPU_CORE_0, RKNN.NPU_CORE_1, RKNN.NPU_CORE_2, RKNN.NPU_CORE_0_1, RKNN.NPU_CORE_0_1_2, RKNN.NPU_CORE_ALL]

MODEL_PATH = "python/models/yolov11n_pose.rknn"

nmsThresh = 0.4
objectThresh = 0.5

def letterbox_resize(image, size, bg_color):        
    if isinstance(image, str):
        image = cv2.imread(image)
    target_width, target_height = size
    image_height, image_width, _ = image.shape
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    innerWidth = xmax - xmin
    innerHeight = ymax - ymin
    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0
    innerArea = innerWidth * innerHeight
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    total = area1 + area2 - innerArea
    return innerArea / total

def NMS(detectResult):
    predBoxs = []
    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process(out, keypoints, index, model_w, model_h, stride, scale_w=1, scale_h=1):
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    out = []
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0, c, (h * model_w) + w] > objectThresh:
                    xywh_ = xywh[0, :, (h * model_w) + w]
                    xywh_ = xywh_.reshape(1, 4, 16, 1)
                    data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)
                    xywh_ = softmax(xywh_, 2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)
                    xywh_temp = xywh_.copy()
                    xywh_temp[0] = (w + 0.5) - xywh_[0]
                    xywh_temp[1] = (h + 0.5) - xywh_[1]
                    xywh_temp[2] = (w + 0.5) + xywh_[2]
                    xywh_temp[3] = (h + 0.5) + xywh_[3]
                    xywh_[0] = ((xywh_temp[0] + xywh_temp[2]) / 2)
                    xywh_[1] = ((xywh_temp[1] + xywh_temp[3]) / 2)
                    xywh_[2] = (xywh_temp[2] - xywh_temp[0])
                    xywh_[3] = (xywh_temp[3] - xywh_temp[1])
                    xywh_ = xywh_ * stride
                    xmin = (xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    keypoint = keypoints[..., (h * model_w) + w + index]
                    keypoint[..., 0:2] = keypoint[..., 0:2] // 1
                    box = DetectBox(c, conf[0, c, (h * model_w) + w], xmin, ymin, xmax, ymax, keypoint)
                    out.append(box)
    return out 

class HumanDetection:
    def __init__(self, target='rk3588', core_mask=RKNN.NPU_CORE_AUTO):
        self.rknn = RKNN(verbose=True)
        self.model_path = MODEL_PATH
        self.target = target
        self.core_mask = core_mask
        self.load_model()
        self.init_runtime()
        self.fps_avg = 0.0
        self.call_count = 0

    def load_model(self):
        try:
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                logger.error(f'Load RKNN model "{self.model_path}" failed!')
                exit(ret)
            logger.info('Model loaded successfully.')
        except Exception as e:
            logger.error(f'An error occurred: {e}')
            exit(1)

    def init_runtime(self):
        try:
            ret = self.rknn.init_runtime(target=self.target, core_mask=self.core_mask)
            if ret != 0:
                logger.error('Init runtime environment failed!')
                exit(ret)
            logger.info('Runtime initialized successfully.')
        except Exception as e:
            logger.error(f'An error occurred: {e}')
            exit(1)

    def preprocess_image(self, img, target_size=(640, 640), pad_value=56):
        letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, target_size, pad_value)
        infer_img = letterbox_img[..., ::-1]  # BGR2RGB
        return infer_img, aspect_ratio, offset_x, offset_y

    def inference(self, infer_img):
        results = self.rknn.inference(inputs=[infer_img], data_format='nhwc')
        return results

    def postprocess(self, results, aspect_ratio, offset_x, offset_y):
        outputs = []
        keypoints = results[3]
        for x in results[:3]:
            index, stride = 0, 0
            if x.shape[2] == 20:
                stride = 32
                index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
            if x.shape[2] == 40:
                stride = 16
                index = 20 * 4 * 20 * 4
            if x.shape[2] == 80:
                stride = 8
                index = 0
            feature = x.reshape(1, 65, -1)
            output = process(feature, keypoints, index, x.shape[3], x.shape[2], stride)
            outputs = outputs + output
        predbox = NMS(outputs)
        for i in range(len(predbox)):
            predbox[i].xmin = int((predbox[i].xmin - offset_x) / aspect_ratio)
            predbox[i].ymin = int((predbox[i].ymin - offset_y) / aspect_ratio)
            predbox[i].xmax = int((predbox[i].xmax - offset_x) / aspect_ratio)
            predbox[i].ymax = int((predbox[i].ymax - offset_y) / aspect_ratio)
            predbox[i].keypoint[..., 0] = (predbox[i].keypoint[..., 0] - offset_x) / aspect_ratio
            predbox[i].keypoint[..., 1] = (predbox[i].keypoint[..., 1] - offset_y) / aspect_ratio
        return predbox

    def draw_results(self, img, predbox):
        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            classId = predbox[i].classId
            score = predbox[i].score
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            ptext = (xmin, ymin)
            title = CLASSES[classId] + "%.2f" % score
            cv2.putText(img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            keypoints = predbox[i].keypoint.reshape(-1, 3)
            for k, keypoint in enumerate(keypoints):
                x, y, conf = keypoint
                color_k = [int(x) for x in kpt_color[k]]
                if x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)
            for k, sk in enumerate(skeleton):
                pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
                pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))
                conf1 = keypoints[(sk[0] - 1), 2]
                conf2 = keypoints[(sk[1] - 1), 2]
                if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                    continue
                cv2.line(img, pos1, pos2, [int(x) for x in limb_color[k]], thickness=2, lineType=cv2.LINE_AA)
        return img

    def run_detection(self, img):
        start_time = time.time()
        infer_img, aspect_ratio, offset_x, offset_y = self.preprocess_image(img)
        results = self.inference(infer_img)
        predbox = self.postprocess(results, aspect_ratio, offset_x, offset_y)
        end_time = time.time()
        duration = end_time - start_time
        fps_current = 1 / duration if duration > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1
        # logger.info(f"FPS Human detection: {self.fps_avg:.2f}")
        keypoints_data = np.array([box.keypoint.reshape(-1, 3)[:, :2] for box in predbox]) if predbox else np.array([])
        boxes_data = [(int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)) for box in predbox]
        return keypoints_data, boxes_data
    
    # def transform_keypoints_to_local(self, box, keypoints):
    #     """Chuyển đổi keypoints từ tọa độ toàn cục sang tọa độ cục bộ trong bounding box."""
    #     xmin, ymin, xmax, ymax = box
    #     local_keypoints = keypoints.copy()
    #     local_keypoints[..., 0] = local_keypoints[..., 0] - xmin
    #     local_keypoints[..., 1] = local_keypoints[..., 1] - ymin
    #     return local_keypoints
    def transform_keypoints_to_local(self, box, keypoints):
        """
        Chuyển đổi keypoints từ ảnh gốc sang tọa độ trong bounding box.
        
        box: Tuple (x1, y1, x2, y2) - Bounding box
        keypoints: List [(x, y), (x, y), ...] - Danh sách keypoints của box này
        
        Returns: List of transformed keypoints [(x', y'), (x', y'), ...]
        """
        x1, y1, _, _ = box
        transformed_keypoints = []

        for (x, y) in keypoints:
            # Chuyển đổi tọa độ
            new_x = x - x1
            new_y = y - y1
            
            # Nếu x hoặc y ban đầu bằng 0, thì x' hoặc y' cũng bằng 0
            if x == 0:
                new_x = 0
            if y == 0:
                new_y = 0

            transformed_keypoints.append((new_x, new_y))

        return transformed_keypoints

    def release(self):
        self.rknn.release()

if __name__ == '__main__':
    target = "rk3588"
    core_mask = RKNN.NPU_CORE_0
    detector = HumanDetection(target, core_mask)
    detector.load_model()
    detector.init_runtime()
    source = cv2.VideoCapture("output_4k_video.mp4")
    while True:
        ret, frame = source.read()
        if not ret:
            logger.error("Không thể đọc frame từ video")
            break
        predbox = detector.detect(frame)
    detector.release()