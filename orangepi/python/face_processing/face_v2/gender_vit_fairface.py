import os
from typing import Union, Tuple, Dict
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_dir = "models/dima806_fairface_gender_image_detection"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    model_dir
)

class GenderClassifier:
    """
    Một lớp để tải mô hình phân loại giới tính và dự đoán giới tính từ hình ảnh,
    sử dụng model 'dima806/fairface_gender_image_detection' fine-tuned trên FairFace.
    """
    def __init__(
        self,
        model_dir: str = MODEL_PATH,
        threshold: float = 0.8,
        device: Union[str, torch.device] = None
    ):
        """
        Args:
            model_dir (str): Tên repo hoặc đường dẫn tới folder chứa model.
            threshold (float): Ngưỡng tin cậy (confidence) để trả về label.
            device (str hoặc torch.device, optional): "cuda" hoặc "cpu". Mặc định tự detect.
        """
        # Thiết lập device
        self.device = torch.device("cpu")

        # Load processor và model từ Hugging Face Hub
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Lấy nhãn từ config luôn cho khớp với quá trình fine-tune
        self.id2label: Dict[int, str] = self.model.config.id2label
        self.threshold = threshold

    def predict(self, image: Union[str, Image.Image]) -> Dict[str, float]:
        """
        Dự đoán xác suất cho từng nhãn từ 1 hình ảnh.

        Args:
            image (str hoặc PIL.Image): đường dẫn tới file hoặc PIL Image.

        Returns:
            Dict[label, probability]
        """
        # Nếu truyền path thì đọc file
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        # Đưa tensor về đúng device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        # Trả về dict label:score
        return {
            self.id2label[i]: float(probs[i].cpu().item())
            for i in range(probs.shape[0])
        }

    def predict_top(self, image: Union[str, Image.Image]) -> Tuple[Union[str, None], float]:
        """
        Trả về nhãn có xác suất cao nhất nếu >= threshold, 
        ngược lại trả về (None, 0.0).

        Returns:
            (label, score)
        """
        preds = self.predict(image)
        top_label, top_score = max(preds.items(), key=lambda x: x[1])
        if top_score >= self.threshold:
            return top_label, top_score
        return None, 0.0
