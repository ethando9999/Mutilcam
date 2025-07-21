import numpy as np
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

class VitAgeClassifier:
    def __init__(self, model_name='nateraw/vit-age-classifier'):
        """
        Khởi tạo VitAgeClassifier
        
        Args:
            model_name (str): Tên model trên Hugging Face Hub
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Load model và feature extractor
        self._load_model()
    
    def _load_model(self):
        """Load model và feature extractor"""
        print(f"Đang tải model: {self.model_name}...")
        try:
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
            
            # Chuyển model sang thiết bị đã chọn (GPU/CPU)
            self.model.to(self.device)
            self.model.eval()  # Đặt model ở chế độ evaluation
            
            print("Tải model và chuyển sang thiết bị thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            raise e
    
    def predict(self, image_array):
        """
        Dự đoán độ tuổi từ mảng numpy
        
        Args:
            image_array (numpy.ndarray): Mảng numpy biểu diễn ảnh
                - Shape có thể là (H, W, 3) cho ảnh RGB
                - Hoặc (H, W) cho ảnh grayscale
                - Giá trị pixel từ 0-255 (uint8) hoặc 0-1 (float)
        
        Returns:
            dict: Dictionary chứa thông tin dự đoán
                - 'predicted_age': Nhóm tuổi được dự đoán
                - 'confidence': Độ tin cậy (probability)
                - 'all_probabilities': Dictionary chứa xác suất cho tất cả các nhóm tuổi
        """
        try:
            # Kiểm tra và chuyển đổi input
            image = self._preprocess_image(image_array)
            
            # Chuẩn bị input cho model
            inputs = self.feature_extractor(images=image, return_tensors='pt')
            
            # Chuyển dữ liệu sang device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Thực hiện dự đoán
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Xử lý kết quả
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Lấy prediction index và label
            prediction_index = logits.argmax(-1).item()
            predicted_label = self.model.config.id2label[prediction_index]
            confidence = probabilities[0][prediction_index].item()
            
            # Tạo dictionary chứa xác suất cho tất cả labels
            all_probabilities = {}
            for idx, prob in enumerate(probabilities[0]):
                label = self.model.config.id2label[idx]
                all_probabilities[label] = prob.item()
            
            return {
                'predicted_age': predicted_label,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
        except Exception as e:
            print(f"Lỗi khi thực hiện dự đoán: {e}")
            raise e
    
    def _preprocess_image(self, image_array):
        """
        Tiền xử lý mảng numpy thành PIL Image
        
        Args:
            image_array (numpy.ndarray): Mảng numpy biểu diễn ảnh
        
        Returns:
            PIL.Image: Ảnh PIL đã được chuyển đổi
        """
        # Chuyển đổi numpy array thành PIL Image
        if isinstance(image_array, np.ndarray):
            # Kiểm tra shape của array
            if len(image_array.shape) == 3:  # (H, W, C)
                if image_array.shape[2] == 3:  # RGB
                    # Nếu giá trị từ 0-1, chuyển về 0-255
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    image = Image.fromarray(image_array, mode='RGB')
                elif image_array.shape[2] == 4:  # RGBA
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
                else:
                    raise ValueError(f"Không hỗ trợ số kênh: {image_array.shape[2]}")
            elif len(image_array.shape) == 2:  # Grayscale (H, W)
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                image = Image.fromarray(image_array, mode='L').convert('RGB')
            else:
                raise ValueError(f"Shape không hợp lệ: {image_array.shape}")
        else:
            raise TypeError("Input phải là numpy array")
        
        return image
    
    def get_all_age_groups(self):
        """
        Lấy danh sách tất cả các nhóm tuổi mà model có thể dự đoán
        
        Returns:
            list: Danh sách các nhóm tuổi
        """
        return list(self.model.config.id2label.values())
    
    def batch_predict(self, image_arrays):
        """
        Dự đoán cho nhiều ảnh cùng lúc
        
        Args:
            image_arrays (list): Danh sách các numpy arrays
        
        Returns:
            list: Danh sách kết quả dự đoán
        """
        results = []
        for image_array in image_arrays:
            result = self.predict(image_array)
            results.append(result)
        return results


# ==============================================================================
# --- CÁCH SỬ DỤNG ---
# ==============================================================================

if __name__ == "__main__":
    # Ví dụ sử dụng
    import numpy as np
    
    # Khởi tạo classifier
    classifier = VitAgeClassifier()
    
    # Tạo ảnh mẫu (thay thế bằng ảnh thật)
    # Ví dụ: ảnh RGB 224x224
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Thực hiện dự đoán
    result = classifier.predict(sample_image)
    
    print("Kết quả dự đoán:")
    print(f"Nhóm tuổi dự đoán: {result['predicted_age']}")
    print(f"Độ tin cậy: {result['confidence']:.2%}")
    print(f"Tất cả xác suất: {result['all_probabilities']}")
    
    # Lấy tất cả nhóm tuổi
    print(f"\nTất cả nhóm tuổi: {classifier.get_all_age_groups()}")