# Hướng dẫn cài đặt Telegram Bot gửi IP

## 1. Tạo Telegram Bot
1. Mở Telegram, chat với [@BotFather](https://t.me/BotFather)
2. Gửi lệnh `/newbot`
3. Nhập tên cho bot (VD: BananaPi IP Bot)
4. Nhập username cho bot (phải kết thúc bằng 'bot', VD: banapi_bot)
5. BotFather sẽ trả về TOKEN của bot (VD: `8133345035:AAGqs4LlKRRyCDxSrahri8w_wWAeHg9OXCs`)
6. Lưu TOKEN này lại

## 2. Tạo Telegram Group
1. Mở Telegram
2. Click "New Message" -> "New Group"
3. Đặt tên group (VD: "BPI BOT")
4. Thêm bot vào group:
   - Click tên group -> "Add members"
   - Tìm và thêm bot (@banapi_bot)
5. Cấp quyền admin cho bot:
   - Click tên group -> "Manage group" -> "Administrators"
   - "Add Admin" -> Chọn bot
   - Bật các quyền cần thiết

## 3. Lấy Group Chat ID
1. Gửi tin nhắn test trong group (VD: "hello")
2. Chạy lệnh để lấy group ID:
```bash
curl https://api.telegram.org/bot8133345035:AAGqs4LlKRRyCDxSrahri8w_wWAeHg9OXCs/getUpdates
```
3. Tìm ID trong kết quả JSON (VD: `-1002356483615`)

## 4. Tạo Script Gửi IP
1. Tạo file script:
```bash
sudo nano /home/bananapi/send_ip.sh
```

2. Thêm nội dung:
```bash
#!/bin/bash

# Đợi network khởi động
sleep 30

# Bot token và group chat id
BOT_TOKEN="8133345035:AAGqs4LlKRRyCDxSrahri8w_wWAeHg9OXCs"
GROUP_CHAT_ID="-1002356483615"

# Lấy hostname
HOSTNAME=$(hostname)

# Lấy thời gian
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Lấy danh sách IP và interface
IP_LIST=$(ip -4 addr show | grep -v "127.0.0.1" | awk '/inet/ {print $NF": "$2}')

# Tạo message
MESSAGE="🖥️ *Device:* \`$HOSTNAME\`
🕒 *Time:* \`$TIMESTAMP\`

*IP Addresses:*"

# Thêm từng IP vào message
while IFS= read -r line; do
    MESSAGE="$MESSAGE
• \`$line\`"
done <<< "$IP_LIST"

# Gửi message vào group
curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
    -d "chat_id=$GROUP_CHAT_ID" \
    -d "text=$MESSAGE" \
    -d "parse_mode=Markdown"

# Log kết quả
echo "[$(date)] Sent IP notification to BPI BOT group" >> /var/log/telegram-ip.log
```

## 5. Cấu hình Permissions
```bash
# Cấp quyền thực thi cho script
sudo chmod +x /home/bananapi/send_ip.sh

# Tạo và phân quyền file log
sudo touch /var/log/telegram-ip.log
sudo chown bananapi:bananapi /var/log/telegram-ip.log
sudo chmod 644 /var/log/telegram-ip.log
```

## 6. Tạo Systemd Service
1. Tạo file service:
```bash
sudo nano /etc/systemd/system/telegram-ip.service
```

2. Thêm nội dung:
```ini
[Unit]
Description=Telegram IP Notifier
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/home/bananapi/send_ip.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

## 7. Kích hoạt Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Kích hoạt service
sudo systemctl enable telegram-ip.service

# Khởi động service
sudo systemctl start telegram-ip.service
```

## 8. Kiểm tra
1. Test script:
```bash
/home/bananapi/send_ip.sh
```

2. Xem log:
```bash
# Log của script
tail -f /var/log/telegram-ip.log

# Log của service
journalctl -u telegram-ip.service -f
```

## 9. Reboot để test
```bash
sudo reboot
```

## Lưu ý
- Bot phải là admin trong group
- Group chat ID phải chính xác
- Đảm bảo network đã online trước khi script chạy
- Service sẽ tự động chạy mỗi khi khởi động
- Có thể xem log để debug nếu có lỗi 