Dưới đây là nội dung **README** dành cho file `/boot/cmdline.txt`, giải thích chi tiết từng tham số:

---

# README: /boot/cmdline.txt

File `/boot/cmdline.txt` trên Raspberry Pi chứa **dòng lệnh kernel** (kernel command line). Nó được sử dụng để cấu hình các tham số cần thiết khi hệ điều hành khởi động.

## Quy tắc chỉnh sửa:
1. Tất cả tham số phải nằm trên **một dòng duy nhất**.
2. Các tham số cách nhau bằng dấu cách (` `).
3. Không được xuống dòng hoặc thêm dòng trống.

---

## Nội dung tham số mẫu:
```txt
console=serial0,115200 console=tty1 root=PARTUUID=9300bc54-02 rootfstype=ext4 fsck.repair=yes rootwait modules-load=dwc2,g_ether cfg80211.ieee80211_regdom=VN
```

---

## Giải thích từng tham số:

### **1. `console=serial0,115200 console=tty1`**
- **Mục đích**: Xác định console để nhận log khởi động từ kernel.
  - `serial0,115200`: Gửi log qua cổng serial với tốc độ 115200 baud.
  - `tty1`: Hiển thị log trên màn hình chính.

### **2. `root=PARTUUID=9300bc54-02`**
- **Mục đích**: Xác định phân vùng chứa hệ điều hành (root filesystem).
- **PARTUUID**: Mã định danh của phân vùng gốc. Phải trùng với phân vùng trong thẻ SD.

### **3. `rootfstype=ext4`**
- **Mục đích**: Chỉ định loại hệ thống tệp của phân vùng gốc (ext4).

### **4. `fsck.repair=yes`**
- **Mục đích**: Kích hoạt kiểm tra và sửa lỗi hệ thống tệp tự động trong quá trình khởi động.

### **5. `rootwait`**
- **Mục đích**: Yêu cầu kernel chờ cho đến khi thiết bị root sẵn sàng trước khi tiếp tục khởi động.

### **6. `modules-load=dwc2,g_ether`**
- **Mục đích**: Nạp các module kernel trong quá trình khởi động.
  - `dwc2`: Kích hoạt chế độ USB OTG (Dual-Role Device Controller).
  - `g_ether`: Kích hoạt chế độ USB Ethernet (RNDIS).

### **7. `cfg80211.ieee80211_regdom=VN`**
- **Mục đích**: Cài đặt vùng pháp lý Wi-Fi.
  - `VN`: Việt Nam.
  - Có thể thay bằng mã quốc gia khác (VD: `GB` cho Vương quốc Anh).

---

## Lưu ý:
- Nếu cần thêm module hoặc tham số mới, hãy đảm bảo dòng lệnh không bị xuống dòng.
- Sau khi chỉnh sửa, khởi động lại Raspberry Pi để áp dụng thay đổi:
  ```bash
  sudo reboot
  ```

---

## Tài liệu tham khảo:
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)