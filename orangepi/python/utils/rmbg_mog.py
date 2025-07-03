import cv2
import numpy as np
import time
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Background Remover Class
class BackgroundRemover:    
    """
    Class for background subtraction using MOG2.
    """
    def __init__(self, history: int = 100, varThreshold: int = 400, learningRate: float = 0.00001): # varThreshold: int = 383
        self.remover = cv2.createBackgroundSubtractorMOG2( 
            history=history,         # Sử dụng nhiều khung hình để cập nhật nền
            varThreshold=varThreshold,  # Ngưỡng thay đổi thấp
            detectShadows=True       # Phát hiện bóng đổ
        )
        self.learning_rate = learningRate
        self.kernel_3 = np.ones((3, 3), np.uint8)
        self.kernel_5 = np.ones((5, 5), np.uint8)

    def update_background(self, frame):
        """
        Cập nhật nền bằng cách áp dụng một khung hình với learning rate mặc định.
        
        Args:
        - image: Khung hình đầu vào.
        """
        logger.info("Updating background...")
        # Resize frame trước khi xử lý 
        frame = self.resize_frame(frame) 
        self.remover.apply(frame, learningRate=self.learning_rate)

    def remove_background(self, frame):
        """
        Remove background in image.
        Args:
        - image: image.
        Return:
        - fgmask: mask of foreground.
        - foreground: foreground extracted.
        - mask_boxes: boxes extracted from foreground.
        """
        # Set learning_rate to 0 to avoid learning during background removal
        self.learning_rate = 0.5
        # Resize frame trước khi xử lý
        frame = self.resize_frame(frame)
        fgmask = self.remover.apply(frame, learningRate=self.learning_rate)
        fgmask = cv2.erode(fgmask, self.kernel_3, iterations=1)
        fgmask = cv2.dilate(fgmask, self.kernel_5, iterations=18)
        fgmask = cv2.medianBlur(fgmask, 5)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:  # Loại bỏ đối tượng nhỏ 
                continue
            box = list(cv2.boundingRect(contour))
            box[2] += box[0]  # Chuyển từ (x, y, w, h) sang (x1, y1, x2, y2)
            box[3] += box[1]
            # Mở rộng box trước khi thêm vào mask_boxes
            expanded_box = self.expand_box(box)
            mask_boxes.append(expanded_box)
        foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
        return fgmask, foreground, mask_boxes
    
    def expand_box(self, box, scale=1.7):
        """
        Mở rộng bounding box theo tỉ lệ scale.

        Parameters:
            box (list): Bounding box dạng [x_min, y_min, x_max, y_max].
            scale (float): Tỉ lệ mở rộng bounding box.

        Returns:
            list: Bounding box đã được mở rộng. 
        """
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        x_min_exp = x_min - width * (scale - 1) / 2
        y_min_exp = y_min - height * (scale - 1) / 2
        x_max_exp = x_max + width * (scale - 1) / 2
        y_max_exp = y_max + height * (scale - 1) / 2
        return [x_min_exp, y_min_exp, x_max_exp, y_max_exp]

    def merge_boxes(self, mask_boxes, scale=1.7, overlap_threshold=0.05):
        """
        Mở rộng và merge các bounding box nếu chúng trùng nhau hơn overlap_threshold.
        Giữ nguyên các box không bị merge.

        Parameters:
            mask_boxes (list of list): Danh sách các bounding box dạng [x_min, y_min, x_max, y_max].
            scale (float): Tỉ lệ mở rộng bounding box.
            overlap_threshold (float): Ngưỡng trùng lặp để merge (mặc định 10%).

        Returns:
            list of list: Danh sách bounding box đã xử lý.
        """
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area

        # Mở rộng các box
        expanded_boxes = []
        for i, box in enumerate(mask_boxes):
            expanded_box = self.expand_box(box, scale)
            expanded_boxes.append(expanded_box + [i])  # Thêm chỉ số gốc

        # Merge các box
        merged_indices = set() 
        final_boxes = []

        while expanded_boxes:
            current_box = expanded_boxes.pop(0)
            to_merge = [current_box]

            for other_box in expanded_boxes[:]:
                if compute_iou(current_box[:4], other_box[:4]) > overlap_threshold:
                    to_merge.append(other_box)
                    expanded_boxes.remove(other_box)

            if len(to_merge) > 1:  # Nếu có trùng lặp, merge lại
                indices = [box[4] for box in to_merge]
                x_min = min(mask_boxes[i][0] for i in indices)
                y_min = min(mask_boxes[i][1] for i in indices)
                x_max = max(mask_boxes[i][2] for i in indices)
                y_max = max(mask_boxes[i][3] for i in indices)
                final_boxes.append([x_min, y_min, x_max, y_max])
                merged_indices.update(indices)
            else:  # Nếu không trùng lặp, giữ nguyên box gốc
                final_boxes.append(mask_boxes[current_box[4]])
                merged_indices.add(current_box[4])

        # Thêm các box chưa được xét đến (không bị merge)
        for i, box in enumerate(mask_boxes):
            if i not in merged_indices:
                final_boxes.append(box)

        return final_boxes
    
    def resize_frame(self, frame, target_width=640):
        """
        Resize frame với width cố định và giữ tỷ lệ khung hình.
        
        Args:
            frame: Khung hình đầu vào
            target_width: Chiều rộng mục tiêu (mặc định 640)
            
        Returns:
            Khung hình đã resize
        """
        height, width = frame.shape[:2]
        self.resize_ratio = target_width / width
        target_height = int(height * self.resize_ratio)
        return cv2.resize(frame, (target_width, target_height))

    def learn_background(self, camerahandler, learning_time=3):
        """
        Learn the background for background removal.

        Args:
            - output: Stream of frames from the camera.
            - background_remover: An instance of BackgroundRemover to update the background model.
            - learning_time (float): Duration for learning the background (in seconds).
        """
        logger.info("Starting background learning...")
        start_time = time.time()

        while time.time() - start_time <= learning_time:
            try:
                frame = camerahandler.capture_lores_frame()
                self.update_background(frame)
            except Exception as e:
                logger.error(f"An error occurred during background learning: {e}")
        logger.info("Background learning completed successfully!")

    def detect(self, frame, scale=1.7, overlap_threshold=0.05):
        """
        Thực hiện remove background và merge boxes để phát hiện đối tượng.

        Parameters:
            frame: Khung hình đầu vào
            scale (float): Tỉ lệ mở rộng bounding box
            overlap_threshold (float): Ngưỡng trùng lặp để merge boxes

        Returns:
            final_boxes: Danh sách các bounding box cuối cùng sau khi đã merge
            fgmask: Mask của foreground
            foreground: Ảnh foreground đã được trích xuất
        """
        # Thực hiện remove background
        fgmask, foreground, mask_boxes = self.remove_background(frame)
        
        # Merge các boxes trùng lặp
        final_boxes = self.merge_boxes(mask_boxes, scale, overlap_threshold)
        
        return fgmask, foreground, final_boxes



    