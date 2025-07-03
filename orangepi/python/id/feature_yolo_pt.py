from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.cfg import get_cfg
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from utils.logging_python_orangepi import get_logger

logger = get_logger(__name__)

current_path = os.getcwd()

# MODEL_PATH = "python/models/yolo11n-pose_ncnn_model" 
MODEL_PATH = "python/models/yolo11n-pose.pt"   

# MODEL_PATH = os.path.join(current_path, MODEL_PATH)

class FeatureExtractorModel(DetectionModel):
    def __init__(self, class_names=['person'], nc=1, verbose=True):
        """
        Initialize the YOLOv11 model with integrated model loading and configuration.

        Args:
            nc (int): Number of classes.
            class_names (list): List of class names.
            verbose (bool): Whether to print model details during initialization.
        """
        try:
            # Load model configuration and weights 
            ckpt = None
            weights, ckpt = attempt_load_one_weight(MODEL_PATH)
            cfg = ckpt['model'].yaml

            # Initialize the DetectionModel with the loaded configuration
            super().__init__(cfg, ch=3, nc=nc, verbose=verbose)

            # Load weights if available
            if weights:
                self.load(weights)

            # Attach class names and hyperparameters to the model
            self.nc = nc
            # self.names = class_names
            self.args = get_cfg(overrides={'model': MODEL_PATH})
            
        except FileNotFoundError as e:
            logger.error(f"Model file không tồn tại: {MODEL_PATH}")
            logger.error(f"Chi tiết lỗi: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo model: {str(e)}")
            raise

    def forward(self, x):
        """
        Forward pass that returns both detection output and features
        from the last convolutional layer before cv2 in the Detect layer.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            features (torch.Tensor): Extracted features from the last convolutional layer.
            detections (torch.Tensor): Detection output.
        """
        try:
            y = []
            features = None
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)  # run the module

                # Capture features before the final processing layer
                if hasattr(m, 'cv2'):
                    features = x  # Store features from the last convolutional layer

                y.append(x if m.i in self.save else None)  # Save output if required
            return features, x  # Return features and detection output
        except Exception as e:
            logger.error(f"Lỗi trong quá trình forward pass: {str(e)}")
            raise

    def extract_feature(self, image: np.ndarray):
        """
        Extract features from an input image (ndarray).

        Args:
            image (np.ndarray): Input image as a numpy array (H, W, C).

        Returns:
            features (torch.Tensor): Extracted features from the model.
            detections (torch.Tensor): Detection output from the model.
        """
        try:
            # Convert the numpy array to a PIL Image
            img = Image.fromarray(image)

            # Preprocess the image
            transform = transforms.Compose([
                transforms.Resize((640, 640)),  # Resize to model input size
                transforms.ToTensor()  # Convert to tensor
            ])
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            # Extract features and detections
            with torch.no_grad():  # Disable gradient calculation for inference
                features, detections = self.forward(img_tensor)

            return features, detections
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất đặc trưng: {str(e)}")
            logger.error(f"Shape của ảnh đầu vào: {image.shape}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize the YOLOv11 model for face detection
    face_detection = FeatureExtractorModel(
        nc=1,
        class_names=['person'],
        verbose=True
    )
    face_detection.eval()  # Set model to evaluation mode

    # Load an image as a numpy array (for example purposes)
    img_path = '/content/person.jpg'
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)  # Convert PIL Image to numpy array

    # Extract features using the new method
    features, detections = face_detection.extract_feature(img_array)

    # Output results
    print("Features shape:", features.shape)
    print("Detections shape:", detections.shape)