import platform
import zipfile
import ast
import numpy as np
import torch

MOEL_PATH = "models/yolo11n-pose_saved_model/yolo11n-pose_float16.tflite"

class TFLiteBackend:
    def __init__(self, model_path=MOEL_PATH, device='cpu', edgetpu=False):
        """
        Khởi tạo TFLiteBackend để load model YOLO ở định dạng TensorFlow Lite.

        Args:
            model_path (str): Đường dẫn đến file model TensorFlow Lite.
            device (str): Thiết bị để đặt output tensor (e.g., 'cpu', 'cuda').
            edgetpu (bool): Nếu True, sử dụng Edge TPU cho inference.
        """
        self.model_path = model_path
        self.torch_device = torch.device(device) 
        self.edgetpu = edgetpu
        self.interpreter = None
        self.input_details = None
        self.output_details = None 
        self.metadata = None

        # Thử import tflite_runtime, nếu không được thì dùng tensorflow.lite
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

        # Load model
        if self.edgetpu:
            # Xử lý Edge TPU
            delegate_lib = {
                "Linux": "libedgetpu.so.1",
                "Darwin": "libedgetpu.1.dylib",
                "Windows": "edgetpu.dll"
            }[platform.system()]
            delegate = load_delegate(delegate_lib)
            self.interpreter = Interpreter(
                model_path=self.model_path,
                experimental_delegates=[delegate]
            )
        else:
            # TFLite thông thường
            self.interpreter = Interpreter(model_path=self.model_path)

        # Cấp phát tensor và lấy thông tin input/output
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Input shape expected by the model:", self.input_details[0]['shape'])

        # Load metadata nếu có
        try:
            with zipfile.ZipFile(self.model_path, "r") as model:
                meta_file = model.namelist()[0]
                self.metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))
        except zipfile.BadZipFile:
            pass

    def forward(self, im):
        """
        Chạy inference trên ảnh đầu vào sử dụng model TFLite.

        Args:
            im (torch.Tensor): Tensor ảnh đầu vào với shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor hoặc list: Kết quả inference.
        """
        # Lấy kích thước ảnh từ input
        h, w = im.shape[2:]
        # Chuyển input sang CPU và numpy
        im_np = im.to('cpu').numpy()

        # Kiểm tra shape mong đợi của model
        expected_shape = self.input_details[0]['shape']
        print("Expected shape:", expected_shape)  # Debug để kiểm tra
        print("Current shape:", im_np.shape)      # Debug để kiểm tra

        # Điều chỉnh thứ tự chiều nếu cần
        if expected_shape[1] == 640 and expected_shape[3] == 3:  # Model mong đợi [1, 640, 640, 3]
            im_np = im_np.transpose(0, 2, 3, 1)  # Từ [1, 3, 640, 640] sang [1, 640, 640, 3]
        elif expected_shape[1] == 3 and expected_shape[2] == 640:  # Model mong đợi [1, 3, 640, 640]
            pass  # Đã đúng định dạng, không cần thay đổi
        else:
            raise ValueError(f"Unexpected input shape from model: {expected_shape}")

        # Xử lý quantization nếu cần
        details = self.input_details[0]
        is_int = details["dtype"] in {np.int8, np.int16}
        if is_int:
            scale, zero_point = details["quantization"]
            im_np = (im_np / scale + zero_point).astype(details["dtype"])

        # Đặt tensor đầu vào và chạy inference
        self.interpreter.set_tensor(details["index"], im_np)
        self.interpreter.invoke()

        # Lấy tất cả các output
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output["index"])
            if is_int:
                scale, zero_point = output["quantization"]
                x = (x.astype(np.float32) - zero_point) * scale
            if x.ndim == 3:
                x[..., [0, 2]] *= w  # x, w
                x[..., [1, 3]] *= h  # y, h
            y.append(x)

        # Chuyển output sang torch tensor
        if len(y) == 1:
            return self.from_numpy(y[0])
        else:
            return [self.from_numpy(x) for x in y]

    def from_numpy(self, x):
        """
        Chuyển numpy array sang torch tensor.

        Args:
            x (np.ndarray): Array cần chuyển đổi.

        Returns:
            torch.Tensor: Tensor đã được chuyển đổi.
        """
        return torch.from_numpy(x).to(self.torch_device) if isinstance(x, np.ndarray) else x