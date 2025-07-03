import torch
from torchvision import transforms
from PIL import Image
from PCB.model import PCB, PCB_test
import os

MODEL_NAME = "net_last.pth"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PCB")

model_path = os.path.join(MODEL_PATH, MODEL_NAME)

class FeatureExtraction:
    def __init__(self, model_path = model_path, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.class_num = 751 # DG‑Market
        # Load trained PCB model
        pcb_model = PCB(class_num=self.class_num)
        pcb_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Wrap with PCB_test for feature extraction
        self.model = PCB_test(pcb_model).to(self.device)
        self.model.eval()

        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # mean
                                 [0.229, 0.224, 0.225])  # std
        ])

    def extract_feature(self, img_path: str) -> torch.Tensor:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Extract feature
        with torch.no_grad():
            feature = self.model(img_tensor)  # shape: [1, 2048, 6]

        # Flatten and normalize
        feature = feature.view(feature.size(0), -1)  # [1, 12288]
        fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(fnorm)

        return feature  # shape: [1, 12288]

# ===========================
# Example usage
if __name__ == '__main__':
    extractor = FeatureExtraction(model_path='pcb_model.pth', class_num=751)
    feat = extractor.extract_feature('example.jpg')
    print('Feature shape:', feat.shape)
