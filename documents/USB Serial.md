# Hướng Dẫn Truyền Dữ Liệu Qua USB Giữa Ubuntu PC và Raspberry Pi

## Mục Tiêu
Hướng dẫn thiết lập và thực hiện truyền dữ liệu giữa Ubuntu PC và Raspberry Pi thông qua kết nối USB sử dụng giao tiếp USB Serial.

---

## Yêu Cầu

### Phần Cứng
- Một máy tính chạy Ubuntu.
- Một Raspberry Pi.
- Dây cáp USB (kết nối từ cổng USB của Raspberry Pi đến cổng USB của Ubuntu PC).

### Phần Mềm
- Python (cài đặt trên cả Ubuntu và Raspberry Pi).
- Driver USB ACM (Ubuntu tự động nhận).

---

## Thiết Lập

### 1. Raspberry Pi
- Kích hoạt chế độ USB Gadget bằng cách chỉnh sửa file `/boot/config.txt`. Thêm dòng sau:
  ```
  dtoverlay=dwc2
  ```
- Chỉnh sửa file `/boot/cmdline.txt`, thêm `modules-load=dwc2,g_serial` ngay sau `rootwait`. Ví dụ:
  ```
  rootwait modules-load=dwc2,g_serial
  ```
- Khởi động lại Raspberry Pi:
  ```bash
  sudo reboot
  ```
- Sau khi khởi động, cổng USB Serial sẽ xuất hiện trên Raspberry Pi tại `/dev/ttyGS0`.

### 2. Ubuntu PC
- Kết nối Raspberry Pi với Ubuntu PC qua dây cáp USB.
- Xác minh cổng USB Serial được nhận diện:
  ```bash
  ls /dev/tty*
  ```
  Bạn sẽ thấy một cổng mới, thường là `/dev/ttyACM0`.

- Đảm bảo quyền truy cập cổng serial:
  ```bash
  sudo chmod 666 /dev/ttyACM0
  ```

---

## Mã Python

### 1. Trên Raspberry Pi
Tạo file Python trên Raspberry Pi để luôn lắng nghe dữ liệu từ Ubuntu PC:

```python
import serial

# Mở cổng USB Serial trên Raspberry Pi
ser = serial.Serial('/dev/ttyGS0', 115200, timeout=1)
print("Listening for data...")

while True:
    data = ser.readline().decode('utf-8').strip()
    if data:
        print(f"Received: {data}")
        ser.write(b"Message received\n")
```

### 2. Trên Ubuntu PC
Tạo file Python trên Ubuntu PC để gửi dữ liệu đến Raspberry Pi:

```python
import serial

# Mở cổng USB Serial trên Ubuntu PC
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# Gửi dữ liệu tới Raspberry Pi
ser.write(b"Hello from Ubuntu\n")

# Nhận phản hồi từ Raspberry Pi
response = ser.readline().decode('utf-8').strip()
print(f"Response: {response}")
```

---

## Thực Hiện
1. Trên Raspberry Pi:
   - Chạy file Python để lắng nghe dữ liệu:
     ```bash
     python3 listener.py
     ```
2. Trên Ubuntu PC:
   - Chạy file Python để gửi dữ liệu:
     ```bash
     python3 sender.py
     ```
3. Quan sát dữ liệu gửi và nhận trên cả hai thiết bị.

---

## Ghi Chú
- Tốc độ baud (115200) phải khớp trên cả hai thiết bị.
- Nếu không nhận được dữ liệu, kiểm tra lại kết nối USB hoặc quyền truy cập cổng serial.
- Có thể dùng `sudo` nếu gặp lỗi quyền.

---

## Tham Khảo
- [Tài liệu Python Serial](https://pythonhosted.org/pyserial/)
- [Cấu hình USB Gadget trên Raspberry Pi](https://www.raspberrypi.org/documentation/configuration/config-txt/)

