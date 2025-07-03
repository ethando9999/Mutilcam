# CẤU HÌNH USB ETHERNET (RNDIS)

## 1. Chuẩn bị trên Raspberry Pi

### 1.1. Kích hoạt USB Gadget Mode
1. **Chỉnh sửa file cấu hình**
   - Mở file `/boot/config.txt`:
     ```bash
     sudo nano /boot/config.txt
     ```
     Thêm dòng sau vào cuối file:
     ```txt
     dtoverlay=dwc2
     ```
   - Mở file `/boot/cmdline.txt`:
     ```bash
     sudo nano /boot/cmdline.txt
     ```
     Thêm `modules-load=dwc2,g_ether` ngay sau `rootwait`.
    ```txt
    console=serial0,115200 console=tty1 root=PARTUUID=9300bc54-02 rootfstype=ext4 fsck.repair=yes rootwait modules-load=dwc2,g_ether cfg80211.ieee80211_regdom=VN
    ```

2. **Tạo script cấu hình USB Ethernet**
   - Tạo file `gadget_setup.sh`:
     ```bash
     sudo nano /usr/local/bin/gadget_setup.sh
     ```
     Thêm nội dung sau vào file:
     ```bash
      #!/bin/bash

      # Tải module cần thiết
      modprobe libcomposite

      # Chuyển đến thư mục USB Gadget
      cd /sys/kernel/config/usb_gadget/

      # Tạo một USB gadget mới
      mkdir -p g1
      cd g1

      # Thiết lập các thông số của USB gadget
      echo 0x1d6b > idVendor    # Vendor ID
      echo 0x0104 > idProduct   # Product ID
      echo 0x0100 > bcdDevice   # Device version
      echo 0x0200 > bcdUSB      # USB 2.0

      # Thiết lập thông tin chuỗi (serial, manufacturer, product)
      mkdir -p strings/0x409
      echo "123456789" > strings/0x409/serialnumber
      echo "FI.AI" > strings/0x409/manufacturer
      echo "Raspberry Pi Zero" > strings/0x409/product

      # Cấu hình thiết lập RNDIS
      mkdir -p configs/c.1/strings/0x409
      echo "Config 1: RNDIS" > configs/c.1/strings/0x409/configuration
      echo 120 > configs/c.1/MaxPower

      # Tạo chức năng RNDIS và liên kết nó với cấu hình
      mkdir -p functions/rndis.usb0
      ln -s functions/rndis.usb0 configs/c.1/

      # Xác định UDC (USB Device Controller)
      ls /sys/class/udc > UDC
     ```
   - Lưu file, rồi cấp quyền thực thi:
     ```bash
     sudo chmod +x /usr/local/bin/gadget_setup.sh
     ```

3. **Khởi chạy script**
   - Chạy lệnh:
     ```bash
     sudo /usr/local/bin/gadget_setup.sh
     ```

### 1.2. Kiểm tra cài đặt
- **Kiểm tra UDC**:
  ```bash
  cat /sys/kernel/config/usb_gadget/g1/UDC
  ls /sys/class/udc
  ```
  - Nếu file chứa tên của một UDC (ví dụ: `20980000.usb`), nghĩa là USB Gadget đã được kích hoạt.
  - Nếu kết quả rỗng:
    - Kiểm tra xem module `dwc2` có được nạp không:
      ```bash
      lsmod | grep dwc2
      ```
      - Nếu không có kết quả, tải module bằng lệnh sau:
        ```bash
        sudo modprobe dwc2
        ```
    - Xóa liên kết cũ (nếu có):
      ```bash
      sudo rm /sys/kernel/config/usb_gadget/g1/configs/c.1/rndis.usb0
      ```
    - Chạy lại script cấu hình:
      ```bash
      sudo /usr/local/bin/gadget_setup.sh
      ```
    - Kiểm tra lại UDC:
      ```bash
      cat /sys/kernel/config/usb_gadget/g1/UDC
      ls /sys/class/udc
      ```
- **Kiểm tra giao diện mạng**:
  - Kiểm tra xem giao diện mạng `usb0` đã được tạo chưa:
    ```bash
    ifconfig
    ```

## 2. Cấu hình địa chỉ IP

### 2.1. Trên Raspberry Pi
- Thiết lập địa chỉ IP cho giao diện mạng `usb0`:
```bash
sudo ifconfig usb0 192.168.7.2 netmask 255.255.255.0 up
```
- Sau khi thiet lap, giao dien `usb0`:
```bash
usb0: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
        inet 192.168.7.2  netmask 255.255.255.0  broadcast 192.168.7.255
        ether aa:84:e9:16:db:af  txqueuelen 1000  (Ethernet)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 0  bytes 0 (0.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

### 2.2. Trên Banana Pi
- Kiểm tra xem Banana Pi có nhận diện cổng USB không:
```bash
dmesg | grep usb
```
ex: 
```bash
[  906.267802] cdc_ether 3-1.3:1.0 usb0: register 'cdc_ether' at usb-sunxi-ehci-1.3, CDC Ethernet Device, 6a:3d:90:ad:fc:f4
``` 
- Thiết lập địa chỉ IP:
```bash
sudo ifconfig usb0 192.168.7.1 netmask 255.255.255.0 up
```
- ifconfig:
```bash
usb0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 192.168.7.1  netmask 255.255.255.0  broadcast 192.168.7.255
        inet6 fe80::683d:90ff:fead:fcf4  prefixlen 64  scopeid 0x20<link>
        ether 6a:3d:90:ad:fc:f4  txqueuelen 1000  (Ethernet)
        RX packets 15  bytes 3420 (3.4 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 8  bytes 676 (676.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
``` 
---
## Test toc do mang dung iperf3:
- install iperf3 trên cả Raspberry Pi và Banana Pi
```bash
sudo apt update
sudo apt install iperf3
```
- Banana Pi làm server:
```bash
iperf3 -s
```
- Trên Raspberry Pi:
```bash
iperf3 -c 192.168.7.1
```



### ✅ **Các bước Tự động thiết lập USB Gadget và địa chỉ IP cho `usb0` khi khởi động trên Raspberry Pi** 🚀  

Dưới đây là hướng dẫn chi tiết và đầy đủ:

---

## 📌 **1. Kích Hoạt USB Gadget Mode** 

### **1.1. Chỉnh sửa cấu hình boot**

**Mở file `/boot/config.txt`:**  
```bash
sudo nano /boot/config.txt
```
Thêm dòng:  
```txt
dtoverlay=dwc2
```

**Mở file `/boot/cmdline.txt`:**  
```bash
sudo nano /boot/cmdline.txt
```
Thêm `modules-load=dwc2,g_ether` ngay sau `rootwait`:  
```txt
... rootwait modules-load=dwc2,g_ether
```

**Khởi động lại Raspberry Pi:**  
```bash
sudo reboot
```

---

## 📌 **2. Tạo Script Cấu Hình USB Gadget**

**Tạo file script gadget_setup.sh:**  
```bash
sudo nano /usr/local/bin/gadget_setup.sh
```

**Thêm nội dung sau:**  
```bash
#!/bin/bash

# Tải module cần thiết
modprobe libcomposite

# Cấu hình USB Gadget
cd /sys/kernel/config/usb_gadget/
mkdir -p g1
cd g1

echo 0x1d6b > idVendor
echo 0x0104 > idProduct
echo 0x0100 > bcdDevice
echo 0x0200 > bcdUSB

mkdir -p strings/0x409
echo "123456789" > strings/0x409/serialnumber
echo "FI.AI" > strings/0x409/manufacturer
echo "Raspberry Pi Zero" > strings/0x409/product

mkdir -p configs/c.1/strings/0x409
echo "Config 1: RNDIS" > configs/c.1/strings/0x409/configuration
echo 120 > configs/c.1/MaxPower

mkdir -p functions/rndis.usb0
ln -s functions/rndis.usb0 configs/c.1/

ls /sys/class/udc > UDC
```

**Cấp quyền thực thi cho script:**  
```bash
sudo chmod +x /usr/local/bin/gadget_setup.sh
```

---

## 📌 **3. Tạo Service Tự Động Thiết Lập USB Gadget**

**Tạo file service:**  
```bash
sudo nano /etc/systemd/system/gadget_setup.service
```

**Thêm nội dung sau:**  
```ini
[Unit]
Description=USB Gadget Setup
After=network.target

[Service]
ExecStart=/usr/local/bin/gadget_setup.sh
Type=oneshot
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

**Kích hoạt service:**  
```bash
sudo systemctl daemon-reload
sudo systemctl enable gadget_setup.service
sudo systemctl start gadget_setup.service
```

**Kiểm tra trạng thái:**  
```bash
sudo systemctl status gadget_setup.service
```

---

## 📌 **4. Tự Động Thiết Lập Địa Chỉ IP Cho `usb0`**

**Tạo file service:**  
```bash
sudo nano /etc/systemd/system/usb0_static_ip.service
```

**Thêm nội dung sau:**  
```ini
[Unit]
Description=Set static IP for usb0
After=gadget_setup.service

[Service]
ExecStart=/sbin/ifconfig usb0 192.168.7.2 netmask 255.255.255.0 up
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

**Kiểm tra trạng thái:**  
```bash
sudo systemctl status usb0_static_ip.service
```

---

## 📌 **5. Kiểm Tra Kết Nối**

1. **Kiểm tra giao diện mạng `usb0`:**  
```bash
ifconfig usb0
```

2. **Ping từ Raspberry Pi đến Banana Pi:**  
```bash
ping 192.168.7.1
```

---

## ✅ **6. Khởi Động Lại Raspberry Pi**

```bash
sudo reboot
```

---

## 🛠️ **Debug (Nếu có lỗi)**  

1. **Kiểm tra log dịch vụ USB Gadget:**  
```bash
sudo systemctl status gadget_setup.service
journalctl -u gadget_setup.service
```

2. **Kiểm tra log dịch vụ IP tĩnh:**  
```bash
sudo systemctl status usb0_static_ip.service
journalctl -u usb0_static_ip.service
```

---

## 🎯 **Kết Quả Cuối Cùng:**  
- Raspberry Pi sẽ tự động thiết lập **USB Gadget**.  
- Giao diện `usb0` sẽ được gán IP tĩnh `192.168.7.2`.  
- Kết nối giữa Raspberry Pi và Banana Pi sẽ hoạt động mỗi khi khởi động.

👉 **Hoàn thành! Giờ bạn có thể khởi động lại và kiểm tra kết nối.** 🚀



