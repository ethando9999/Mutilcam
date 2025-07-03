# Hướng Dẫn Cài Đặt PiShrink và Sao Chép Thẻ Nhớ từ 64.1GB sang 63.9GB trên Ubuntu

## 1. Giới Thiệu
PiShrink là một script giúp thu nhỏ file image của Raspberry Pi, cho phép sao chép từ một thẻ nhớ lớn hơn sang một thẻ nhớ nhỏ hơn một cách hiệu quả. Trong hướng dẫn này, chúng ta sẽ sử dụng **PiShrink** để thu nhỏ file image và **Balena Etcher** để ghi image vào thẻ nhớ mới.

## 2. Cài Đặt PiShrink
Để sử dụng PiShrink, bạn cần cài đặt nó trên Ubuntu bằng các lệnh sau:

```bash
sudo apt update
sudo apt install git -y
git clone https://github.com/Drewsif/PiShrink.git
cd PiShrink
chmod +x pishrink.sh
sudo cp pishrink.sh /usr/local/bin/
```

Sau khi cài đặt, kiểm tra xem PiShrink đã hoạt động chưa bằng cách chạy:

```bash
pishrink.sh -h
```

## 3. Tạo File Image từ Thẻ Nhớ Cũ (64.1GB)
Giả sử thẻ nhớ cũ có thiết bị là `/dev/sdX`, bạn cần tạo một bản sao image của nó trước khi thu nhỏ:

```bash
sudo dd if=/dev/sdX of=raspbian.img bs=4M status=progress
```

> **Lưu ý:** Thay `sdX` bằng tên thực tế của thiết bị. Bạn có thể kiểm tra bằng lệnh:
>
> ```bash
> lsblk
> sudo fdisk -l
> ```

## 4. Thu Nhỏ Image bằng PiShrink
Sau khi tạo file image, sử dụng PiShrink để thu nhỏ nó:

```bash
sudo pishrink.sh raspbian.img
```

Quá trình này sẽ tự động giảm dung lượng của file image xuống mức thấp nhất có thể.

## 5. Ghi Image Đã Thu Nhỏ Sang Thẻ Nhớ Mới (63.9GB)
### 5.1 Cài đặt Balena Etcher
Truy cập trang web chính thức của **Balena Etcher** tại:

[https://etcher.balena.io/#download-etcher](https://etcher.balena.io/#download-etcher)

Tải xuống phiên bản file **ZIP** dành cho Linux - Ubuntu.

Giải nén và chạy Balena Etcher bằng lệnh:

```bash
unzip balena-etcher-electron.zip
cd balena-etcher-electron
./balena-etcher-electron
```

### 5.2 Chép Image Vào Thẻ Nhớ
1. Nhấn **"Flash from file"** và chọn file `raspbian.img` đã thu nhỏ bằng PiShrink.
2. Nhấn **"Select target"**, chọn thẻ nhớ mới (63.9GB).
3. Nhấn **"Flash!"** để bắt đầu quá trình ghi.
4. Chờ đến khi quá trình hoàn tất, Balena Etcher sẽ tự động kiểm tra lại file (Verify) để đảm bảo ghi thành công.

## 6. Kiểm Tra Thẻ Nhớ Sau Khi Ghi
Sau khi hoàn thành, bạn có thể kiểm tra thẻ nhớ mới bằng cách:

```bash
lsblk
```

Hoặc lắp vào Raspberry Pi để kiểm tra xem hệ thống có hoạt động bình thường không.

## 7. Kết Luận
Với các bước trên, bạn đã có thể thu nhỏ và sao chép thành công một thẻ nhớ Raspberry Pi từ 64.1GB sang 63.9GB trên Ubuntu một cách an toàn và hiệu quả.

Chúc bạn thành công!

