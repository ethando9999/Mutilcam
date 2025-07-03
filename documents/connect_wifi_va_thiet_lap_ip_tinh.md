Dưới đây là các bước chi tiết để kết nối WiFi, thiết lập địa chỉ IP tĩnh và kiểm tra kết quả sử dụng `nmcli` trên Banana Pi:

### **Bước 1: Kết nối WiFi "meee"**
1. Đầu tiên, sử dụng lệnh sau để kết nối vào mạng WiFi "meee" với mật khẩu "abcD1234":
   ```bash
   sudo nmcli dev wifi connect "meee" password "abcD1234"
   ```

   Sau khi lệnh này chạy thành công, bạn sẽ kết nối với mạng WiFi "meee".

### **Bước 2: Kiểm tra tên kết nối WiFi**
2. Xác định tên kết nối WiFi của bạn bằng lệnh sau:
   ```bash
   nmcli connection show
   ```
   Bạn sẽ thấy danh sách các kết nối hiện có. Hãy chú ý đến tên của kết nối WiFi (nó có thể là `wlan0`, `Wi-Fi`, hoặc một tên khác tùy thuộc vào cấu hình hệ thống).

### **Bước 3: Cấu hình IP tĩnh**
3. Cấu hình địa chỉ IP tĩnh cho kết nối WiFi:
   ```bash
   sudo nmcli connection modify "meee" ipv4.addresses 192.168.2.100/24 ipv4.gateway 192.168.2.1 ipv4.dns "8.8.8.8 8.8.4.4"
   sudo nmcli connection modify "meee" ipv4.method manual
   ```
   **Lưu ý**: Thay `"your_connection_name"` bằng tên kết nối WiFi của bạn từ bước trước. Ví dụ: nếu tên kết nối là `wlan0`, bạn thay `"your_connection_name"` bằng `wlan0`.

   Các thông số:
   - `ipv4.addresses 192.168.2.100/24`: Địa chỉ IP tĩnh bạn muốn cấu hình.
   - `ipv4.gateway 192.168.2.1`: Địa chỉ IP của router/gateway.
   - `ipv4.dns "8.8.8.8 8.8.4.4"`: Địa chỉ DNS (Google DNS).

### **Bước 4: Khởi động lại kết nối**
4. Sau khi cấu hình xong, bạn cần khởi động lại kết nối để áp dụng thay đổi:
   ```bash
   sudo nmcli connection down "meee"
   sudo nmcli connection up "meee"
   ```

### **Bước 5: Kiểm tra IP**
5. Kiểm tra lại địa chỉ IP đã được áp dụng bằng lệnh:
   ```bash
   ip a
   ```
   hoặc:
   ```bash
   hostname -I
   ```
   Bạn sẽ thấy địa chỉ IP `192.168.2.100` (hoặc IP mà bạn đã cấu hình) xuất hiện trong danh sách địa chỉ IP của hệ thống.

### **Bước 6: Kiểm tra kết nối mạng**
6. Kiểm tra kết nối đến gateway và DNS:
   - Ping gateway để kiểm tra kết nối mạng:
     ```bash
     ping 192.168.2.1
     ```
   - Ping một trang web để kiểm tra khả năng truy cập Internet:
     ```bash
     ping google.com
     ```

   Nếu cả hai lệnh trên đều thành công, có nghĩa là kết nối mạng và cấu hình IP tĩnh của bạn đã hoạt động đúng.

### **Kết quả mong đợi:**
- Bạn sẽ kết nối thành công với mạng WiFi "meee".
- Địa chỉ IP tĩnh `192.168.2.100` sẽ được áp dụng.
- Kết nối mạng và truy cập Internet sẽ hoạt động như mong đợi.

Nếu có vấn đề gì xảy ra trong các bước trên, bạn có thể kiểm tra lại cấu hình hoặc kết nối WiFi để đảm bảo mọi thứ hoạt động đúng.