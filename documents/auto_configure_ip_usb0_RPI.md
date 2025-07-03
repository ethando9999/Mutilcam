### ✅ **Tự động thiết lập địa chỉ IP cho `usb0` trên Banana Pi**

Dưới đây là các bước để tự động thiết lập địa chỉ IP cho `usb0` khi Banana Pi khởi động:

---

## 📌 **1. Tạo Script Thiết Lập IP Tĩnh Cho `usb0`**

**Tạo file script `usb0_static_ip.sh`:**  
```bash
sudo nano /usr/local/bin/usb0_static_ip.sh
```

**Thêm nội dung sau vào file:**  
```bash
#!/bin/bash

# Thiết lập địa chỉ IP tĩnh cho usb0
/sbin/ifconfig usb0 192.168.7.1 netmask 255.255.255.0 up
```

**Cấp quyền thực thi cho script:**  
```bash
sudo chmod +x /usr/local/bin/usb0_static_ip.sh
```

---

## 📌 **2. Tạo Service Systemd Để Chạy Script Khi Khởi Động**

**Tạo file service `usb0_static_ip.service`:**  
```bash
sudo nano /etc/systemd/system/usb0_static_ip.service
```

**Thêm nội dung sau:**  
```ini
[Unit]
Description=Set static IP for usb0
After=network.target

[Service]
ExecStart=/usr/local/bin/usb0_static_ip.sh
Type=oneshot
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

**Kích hoạt service:**  
```bash
sudo systemctl daemon-reload
sudo systemctl enable usb0_static_ip.service
sudo systemctl start usb0_static_ip.service
```

**Kiểm tra trạng thái service:**  
```bash
sudo systemctl status usb0_static_ip.service
```

---

## 📌 **3. Kiểm Tra Kết Nối Mạng**

1. **Kiểm tra giao diện `usb0`:**  
```bash
ifconfig usb0
```

2. **Ping từ Banana Pi đến Raspberry Pi:**  
```bash
ping 192.168.7.2
```

---

## 📌 **4. Khởi Động Lại Banana Pi**

```bash
sudo reboot
```

---

## 🎯 **Kết Quả Cuối Cùng:**
- Banana Pi sẽ tự động thiết lập IP tĩnh cho `usb0` khi khởi động (`192.168.7.1`).
- Kết nối giữa Raspberry Pi và Banana Pi sẽ được tự động cấu hình.

---

👉 **Giờ bạn có thể kiểm tra kết nối sau khi khởi động lại Banana Pi.** 🚀
