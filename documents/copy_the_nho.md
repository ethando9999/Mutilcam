# Hướng dẫn sao lưu thẻ nhớ Raspberry Pi

## 1️⃣ Xóa block trống để tối ưu dung lượng ảnh `.img`
Trước khi sao lưu, cần làm sạch các block trống để tránh sao chép dữ liệu rác.

```bash
sudo apt install zerofree  # Cài đặt zerofree nếu chưa có
sudo zerofree /dev/sdXn    # Thay sdXn bằng phân vùng cần làm sạch (ví dụ: /dev/sda2)
```

## 2️⃣ Sao lưu chỉ phần cần thiết
Dùng `dd` để sao lưu nhưng tối ưu bằng `sparse` để giảm dung lượng file ảnh.

```bash
sudo dd if=/dev/sdX of=raspberry_backup.img bs=4M status=progress conv=sparse
```
Thay `sdX` bằng tên thiết bị thẻ nhớ, ví dụ: `/dev/sda`.

## 3️⃣ Điều chỉnh kích thước file ảnh
Sau khi sao lưu, nếu cần giảm kích thước file `.img` về dung lượng mong muốn:

```bash
sudo truncate -s 52G raspberry_backup.img  # Điều chỉnh kích thước file ảnh
```

## Lưu ý
- Kiểm tra đúng tên thiết bị (`lsblk` hoặc `fdisk -l`) trước khi thực hiện.
- Sao lưu phân vùng quan trọng trước khi thao tác.
- `zerofree` chỉ hoạt động trên phân vùng `ext2/3/4` và phải được gắn ở chế độ `read-only`.

## Khôi phục ảnh đã sao lưu
Khi cần ghi ảnh `.img` trở lại thẻ nhớ:

```bash
sudo dd if=raspberry_backup.img of=/dev/sdX bs=4M status=progress
```

Thay `sdX` bằng thiết bị thẻ nhớ cần ghi.

Ngoài ra, có thể sử dụng **Balena Etcher** để ghi file ảnh `.img` vào thẻ nhớ một cách dễ dàng:
1. Tải và cài đặt Balena Etcher từ [balena.io/etcher](https://www.balena.io/etcher/).
2. Mở Balena Etcher, chọn **Flash from file**, duyệt đến file `raspberry_backup.img`.
3. Chọn thẻ nhớ đích và nhấn **Flash!** để bắt đầu ghi.

