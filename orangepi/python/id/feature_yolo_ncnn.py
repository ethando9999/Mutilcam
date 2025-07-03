import ncnn
import numpy as np
from ultralytics.nn.tasks import DetectionModel

class FeatureExtractorModel(DetectionModel):
    def __init__(self, class_names=['person'], nc=1, verbose=True):
        """
        Initialize the NCNN YOLO model.
        """
        self.net = ncnn.Net()
        
        # Load NCNN model files 
        param_path = "python/models/yolo11n-pose_ncnn_model/model.param"
        bin_path = "python/models/yolo11n-pose_ncnn_model/model.bin"
        
        # Load model vào NCNN
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # Khởi tạo các thông số khác
        self.nc = nc 
        self.class_names = class_names
        
    def forward(self, x): 
        """
        Thực hiện inference với NCNN model
        x: input image as numpy array
        """
        # Tạo NCNN extractor
        with self.net.create_extractor() as ex:
            # Đặt input
            ex.input("input", ncnn.Mat(x))
            
            # Lấy output
            ret, out = ex.extract("output")
            
            return out.numpy()