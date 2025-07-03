import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image


model_dir = "models/prithivMLmods_gender_classifier_mini"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    model_dir
)


class GenderClassifier:
    """
    Một lớp để tải mô hình phân loại giới tính và dự đoán giới tính từ hình ảnh.
    """
    def __init__(self, model_dir=MODEL_PATH, threshold=0.8):
        """
        Khởi tạo bộ xử lý và mô hình.

        Args:
            model_dir (str): Thư mục chứa các tệp mô hình đã tải xuống.
            threshold (float): Ngưỡng tin cậy để trả về kết quả dự đoán.
        """
        # Tải bộ tiền xử lý ảnh và mô hình từ thư mục đã lưu
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        
        # Lấy nhãn từ file cấu hình của mô hình
        # self.id2label = self.model.config.id2label
        self.id2label = {0: "Male", 1: "Female"}
        self.threshold = threshold

    def predict(self, image):
        """
        Dự đoán tất cả các xác suất cho các nhãn từ một hình ảnh.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            predictions = {self.id2label[i]: round(probs[i].item(), 3) for i in range(len(probs))}
            
        return predictions

    def predict_top(self, image):
        """
        Dự đoán và chỉ trả về nhãn có xác suất cao nhất nếu nó vượt qua ngưỡng.
        """
        predictions = self.predict(image)
        top_gender = max(predictions, key=predictions.get)
        
        if predictions[top_gender] >= self.threshold:
            return top_gender, predictions[top_gender]
            
        return None, 0.0
