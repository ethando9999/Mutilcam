Để tự động chạy các lệnh trên khi khởi động hệ thống, bạn có thể tạo một script khởi động bằng `systemd`:

### 1. Tạo file service  
```bash
sudo nano /etc/systemd/system/wifi-setup.service
```

### 2. Thêm nội dung sau:  
```ini
[Unit]
Description=WiFi Setup Service
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/systemctl enable networking
ExecStart=/bin/systemctl enable hostapd
ExecStart=/bin/systemctl enable dnsmasq
ExecStart=/bin/systemctl restart networking
ExecStart=/bin/systemctl restart hostapd
ExecStart=/bin/systemctl restart dnsmasq
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

### 3. Kích hoạt service  
```bash
sudo systemctl daemon-reload
sudo systemctl enable wifi-setup.service
sudo systemctl start wifi-setup.service
```

### 4. Kiểm tra trạng thái  
```bash
sudo systemctl status wifi-setup.service
```

Bây giờ các lệnh trên sẽ tự động chạy mỗi khi khởi động hệ thống.