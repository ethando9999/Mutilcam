import os
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
from PIL import Image

model_dir = "models/prithivMLmods_open-age-detection"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    model_dir
)

class AgePredictor:
    def __init__(self, model_dir=MODEL_PATH, threshold=0.8):
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = SiglipForImageClassification.from_pretrained(model_dir)
        # self.id2label = {0: "Child 0-12", 1: "Teenager 13-20", 2: "Adult 21-44", 3: "Middle Age 45-64", 4: "Aged 65+"}
        self.id2label = {0: "Child", 1: "Teenager", 2: "Adult", 3: "Middle Age", 4: "Aged"}
        self.threshold = threshold

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            predictions = {self.id2label[i]: round(probs[i].item(), 3) for i in range(len(probs))}
        return predictions

    def predict_top(self, image):
        predictions = self.predict(image)
        top_age = max(predictions, key=predictions.get)
        if predictions[top_age] >= self.threshold:
            return top_age, predictions[top_age]
        return None, 0.0  # Giá trị mặc định thay vì None