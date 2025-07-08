# Hướng dẫn Cài đặt Môi trường AI trên Orange Pi để Sửa lỗi và Chạy Ứng dụng

## Giới thiệu

Tài liệu này cung cấp một quy trình cài đặt đầy đủ và chi tiết để thiết lập môi trường Python cho các ứng dụng AI/Computer Vision phức tạp trên các thiết bị Orange Pi sử dụng chip Rockchip (có NPU).

Việc cài đặt không đúng thứ tự hoặc thiếu các công cụ hệ thống thường dẫn đến các lỗi khó chẩn đoán như `ModuleNotFoundError`, xung đột phiên bản, hoặc thậm chí là lỗi nghiêm trọng như `Segmentation Fault` khi chạy các thư viện được biên dịch từ C++. Hướng dẫn này được thiết kế để giải quyết triệt để các vấn đề đó.

---

## Bước 1: Cài đặt Các Công cụ Hệ thống Nền tảng

Đây là bước nền tảng quan trọng nhất. Chúng ta cần cài đặt các công cụ biên dịch (`build-essential`, `cmake`) và các thư viện toán học hiệu năng cao (`OpenBLAS`, `OpenMP`) mà các gói Python sau này sẽ cần để tự biên dịch.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libomp-dev python3-dev
```

---

## Bước 2: Tạo và Kích hoạt Môi trường ảo Python

Luôn sử dụng môi trường ảo (`venv`) để quản lý các gói phụ thuộc của dự án một cách độc lập, tránh xung đột với các gói hệ thống.

```bash
# Cài đặt công cụ tạo môi trường ảo (nếu chưa có)
sudo apt-get install -y python3-venv

# Tạo một môi trường ảo có tên là 'venv' trong thư mục hiện tại
python3 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate
```
Sau khi kích hoạt, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh. Mọi lệnh `pip` sau này sẽ chỉ tác động bên trong môi trường này.

---

## Bước 3: Cài đặt Các Thư viện Python Cơ bản

Cài đặt tất cả các thư viện Python tiêu chuẩn mà ứng dụng yêu cầu. Gộp chúng vào một lệnh duy nhất để `pip` tự giải quyết các phiên bản phụ thuộc một cách hiệu quả nhất.

```bash
pip install psutil aiosqlite opencv-python torchreid scipy gdown tensorboard scikit-learn faiss-cpu mediapipe dlib transformers aio_pika orjson
```

---

## Bước 4: Cài đặt RKNN-Toolkit2 (Phần Quan trọng nhất)

Đây là bước quyết định để tận dụng sức mạnh của NPU trên chip Rockchip. Thư viện này **không có trên PyPI** và phải được tải về thủ công từ GitHub.

### LƯU Ý ĐẶC BIỆT: Hai lệnh cần chú ý kỹ

Hai lệnh sau đây là chìa khóa để cài đặt thành công. Bạn phải đảm bảo chúng được thực hiện chính xác.

**1. Tải về gói RKNN-Toolkit2 bằng `wget`**

Lệnh này sẽ tải về tệp cài đặt (`.whl`) phù hợp với kiến trúc (`aarch64`) và phiên bản Python của bạn.

*   **Kiểm tra phiên bản Python:** Chạy lệnh `python3 --version`.
*   **Giải thích tên tệp:** Tên tệp `...-cp310-...` dành cho **Python 3.10**. Nếu bạn dùng phiên bản khác (ví dụ Python 3.8), bạn cần tìm tệp có `cp38` trên [trang GitHub của airockchip](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages/arm64) và sửa lại URL trong lệnh `wget` dưới đây.

**Với Python 3.10, hãy chạy lệnh sau:**
```bash
wget https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```
> **Cảnh báo Quan trọng:** Luôn đảm bảo URL chứa `/raw/` chứ không phải `/blob/`. URL chứa `/blob/` sẽ chỉ tải về một trang web HTML, không phải tệp thư viện thực sự, và sẽ gây lỗi khi cài đặt.

**2. Cài đặt tệp `.whl` vừa tải về**

Sau khi `wget` tải xong, hãy dùng `pip` để cài đặt tệp này. Tên tệp phải **chính xác tuyệt đối** với tên tệp bạn vừa tải về.

```bash
pip install rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```
> **Mẹo:** Bạn có thể gõ `pip install rknn` rồi nhấn phím `Tab` trên bàn phím để shell tự động điền nốt tên tệp, tránh gõ sai.

---

## Bước 5: Đồng bộ hóa Phiên bản và Sửa lỗi Xung đột

Việc cài đặt `rknn-toolkit2` có các yêu cầu phiên bản rất khắt khe và có thể đã hạ cấp một số thư viện bạn đã cài trước đó (ví dụ: `torch`, `protobuf`). Điều này gây ra lỗi xung đột.

Hãy chạy lệnh sau để `pip` tự động tìm và cài đặt các phiên bản của `torchvision`, `torchaudio`, và `mediapipe` sao cho chúng tương thích với môi trường hiện tại.

```bash
pip install --upgrade torchvision torchaudio mediapipe
```

---

## Bước 6: Chạy Ứng dụng

Môi trường của bạn bây giờ đã hoàn tất, đồng bộ và sẵn sàng. Bạn có thể chạy ứng dụng của mình.

```bash
python3 python/run_new.py --new-db --device-id opi
```

Chúc mừng! Bạn đã cài đặt thành công một môi trường AI phức tạp và được tối ưu hóa cho phần cứng.