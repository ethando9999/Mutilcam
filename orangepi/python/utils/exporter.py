import torch
from pathlib import Path
import subprocess
import shutil
import os
import logging as logger
import json
import yaml

class Exporter:
    def __init__(self, model, file, args, device, metadata):
        """
        Khởi tạo class Exporter.
        
        Args:
            model: Mô hình PyTorch cần xuất (ví dụ: osnet_ain_x1_0).
            file: Đường dẫn đến file mô hình gốc (Path object).
            args: Đối tượng chứa các tham số cấu hình (optimize, half, nms).
            device: Thiết bị chạy mô hình (CPU hoặc GPU).
            metadata: Thông tin metadata của mô hình (dict).
        """
        self.model = model
        self.file = Path(file)
        self.args = args
        self.device = device
        self.metadata = metadata

    def export_torchscript(self, prefix="TorchScript:"):
        """
        Xuất mô hình sang định dạng TorchScript.
        
        Args:
            prefix (str): Tiền tố để hiển thị log.
        
        Returns:
            tuple: (đường dẫn file TorchScript, None).
        """
        logger.info(f"\n{prefix} starting export with torch {torch.__version__}...")
        f = self.file.with_suffix(".torchscript")

        # Tạo tensor đầu vào giả với kích thước [1, 3, 256, 128] cho osnet_ain_x1_0
        dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
        self.model.eval()  # Đặt mô hình ở chế độ đánh giá
        ts = torch.jit.trace(self.model, dummy_input, strict=False)  # Trace mô hình
        
        # Thêm metadata
        extra_files = {"config.txt": json.dumps(self.metadata)}
        
        if self.args.optimize:  # Tối ưu hóa cho mobile nếu cần
            logger.info(f"{prefix} optimizing for mobile...")
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        
        logger.info(f"{prefix} export success, saved as {f}")
        return f, None

    def export_ncnn(self, prefix="NCNN:"):
        """
        Xuất mô hình sang định dạng NCNN bằng PNNX.
        
        Args:
            prefix (str): Tiền tố để hiển thị log.
        
        Returns:
            tuple: (đường dẫn thư mục NCNN, None).
        """
        import ncnn  # noqa
        logger.info(f"\n{prefix} starting export with NCNN {ncnn.__version__}...")
        f = Path(str(self.file).replace(self.file.suffix, f"_ncnn_model{os.sep}"))
        f_ts = self.file.with_suffix(".torchscript")

        # Kiểm tra PNNX binary
        name = Path("pnnx.exe" if os.name == 'nt' else "pnnx")
        pnnx = name if name.is_file() else (Path.cwd() / name)
        if not pnnx.is_file():
            logger.warning(f"{prefix} WARNING ⚠️ PNNX not found. Please ensure PNNX binary is available.")
            raise FileNotFoundError("PNNX binary not found. Download from https://github.com/pnnx/pnnx/")

        # Thiết lập tham số NCNN và PNNX
        ncnn_args = [
            f"ncnnparam={f / 'model.ncnn.param'}",
            f"ncnnbin={f / 'model.ncnn.bin'}",
            f"ncnnpy={f / 'model_ncnn.py'}",
        ]
        pnnx_args = [
            f"pnnxparam={f / 'model.pnnx.param'}",
            f"pnnxbin={f / 'model.pnnx.bin'}",
            f"pnnxpy={f / 'model_pnnx.py'}",
            f"pnnxonnx={f / 'model.pnnx.onnx'}",
        ]
        inputshape = [1, 3, 256, 128]  # Kích thước đầu vào cho osnet_ain_x1_0

        # Xây dựng lệnh PNNX
        cmd = [
            str(pnnx),
            str(f_ts),
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(self.args.half)}",
            "device=gpu" if self.device.type == "cuda" else "device=cpu",
            f'inputshape="{inputshape}"',
        ]
        
        # Tạo thư mục và chạy lệnh
        f.mkdir(exist_ok=True)
        logger.info(f"{prefix} running '{' '.join(cmd)}'")
        subprocess.run(cmd, check=True)

        # Lưu metadata
        with open(f / "metadata.yaml", 'w') as yaml_file:
            yaml.dump(self.metadata, yaml_file)
        
        logger.info(f"{prefix} export success, saved as {f}")
        return str(f), None
    
if __name__ == "__main__":
    # Tải mô hình osnet_ain_x1_0
    import torchreid
    model = torchreid.models.build_model(name='osnet_ain_x1_0', num_classes=1000, pretrained=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    # Thiết lập tham số
    file = Path("osnet_ain_x1_0.pt")
    args = type("Args", (), {"optimize": True, "half": False, "nms": False})()
    metadata = {"model": "osnet_ain_x1_0", "input_size": [256, 128]}

    # Khởi tạo Exporter
    exporter = Exporter(model, file, args, device, metadata)

    # Xuất TorchScript
    torchscript_file, _ = exporter.export_torchscript()

    # Xuất NCNN
    ncnn_dir, _ = exporter.export_ncnn()