Dưới đây là nội dung tài liệu .md hướng dẫn cách chuyển định dạng video camera sang MJPEG trên hệ thống Linux nhúng (ví dụ: Ubuntu trên MTK-Genio) dựa trên thông tin bạn cung cấp:
Hướng dẫn chuyển định dạng video camera sang MJPEG
Giới thiệu
Tài liệu này hướng dẫn cách cấu hình camera trên hệ thống Linux để chuyển sang định dạng MJPEG (Motion-JPEG) bằng công cụ v4l2-ctl. MJPEG là một định dạng video nén, trong đó mỗi khung hình được nén riêng lẻ dưới dạng JPEG, phù hợp cho các ứng dụng yêu cầu độ trễ thấp hoặc xử lý đơn giản.
Yêu cầu
Hệ điều hành Linux (ví dụ: Ubuntu) có cài đặt công cụ v4l2-ctl (thuộc gói v4l-utils).
Thiết bị camera hỗ trợ định dạng MJPEG (trong ví dụ này là /dev/video1 và /dev/video7).
Các bước thực hiện
1. Kiểm tra thiết bị camera
Trong hệ thống Linux, camera được đại diện bởi các tệp thiết bị như /dev/video0, /dev/video1, v.v. Để liệt kê các thiết bị camera có sẵn, chạy lệnh:
bash
ls /dev/video*
Ví dụ này sử dụng /dev/video1 và /dev/video7.
2. Kiểm tra các định dạng được hỗ trợ
Sử dụng lệnh sau để liệt kê các định dạng mà camera hỗ trợ:
bash
v4l2-ctl --device=/dev/video1 --list-formats
v4l2-ctl --device=/dev/video7 --list-formats
Kết quả mẫu:
Đối với /dev/video1:
Type: Video Capture
[0]: 'MJPG' (Motion-JPEG, compressed)
[1]: 'YUYV' (YUYV 4:2:2)
Đối với /dev/video7:
Type: Video Capture
[0]: 'YUYV' (YUYV 4:2:2)
[1]: 'MJPG' (Motion-JPEG, compressed)
Cả hai thiết bị đều hỗ trợ định dạng 'MJPG' (MJPEG).
3. Kiểm tra định dạng hiện tại
Để xem định dạng hiện tại của camera, sử dụng lệnh:
bash
v4l2-ctl --device=/dev/video1 --get-fmt-video
v4l2-ctl --device=/dev/video7 --get-fmt-video
Kết quả mẫu:
Đối với /dev/video1:
Pixel Format      : 'YUYV' (YUYV 4:2:2)
Width/Height      : 1920/1080
Đối với /dev/video7:
Pixel Format      : 'YUYV' (YUYV 4:2:2)
Width/Height      : 1280/720
Hiện tại, cả hai camera đều đang sử dụng định dạng 'YUYV'.
4. Chuyển định dạng sang MJPEG
Để chuyển định dạng video sang MJPEG, sử dụng lệnh sau:
bash
v4l2-ctl --device=/dev/video1 --set-fmt-video=pixelformat=MJPG
v4l2-ctl --device=/dev/video7 --set-fmt-video=pixelformat=MJPG
Lưu ý quan trọng:
Sử dụng 'MJPG' (4 ký tự) thay vì 'MJPEG'. Nếu sử dụng 'MJPEG', bạn sẽ gặp lỗi:
The pixelformat 'MJPEG' is invalid
5. Xác nhận thay đổi
Kiểm tra lại định dạng sau khi cấu hình:
    bash
v4l2-ctl --device=/dev/video1 --get-fmt-video
v4l2-ctl --device=/dev/video7 --get-fmt-video
Kết quả mong muốn:
Đối với /dev/video1:
Pixel Format      : 'MJPG' (Motion-JPEG, compressed)
Width/Height      : 1920/1080
Đối với /dev/video7:
Pixel Format      : 'MJPG' (Motion-JPEG, compressed)
Width/Height      : 1280/720
Nếu kết quả hiển thị 'MJPG', cấu hình đã thành công.
Giải quyết vấn đề thường gặp
Lỗi "The pixelformat 'MJPEG' is invalid":
Nguyên nhân: Sử dụng sai mã định dạng.
Khắc phục: Thay 'MJPEG' bằng 'MJPG'.
Camera không hỗ trợ MJPEG:
Kiểm tra lại bước 2. Nếu 'MJPG' không có trong danh sách, camera không hỗ trợ định dạng này.
Lệnh v4l2-ctl không hoạt động:
Cài đặt v4l-utils bằng lệnh:
bash
sudo apt-get install v4l-utils
Kết luận
Việc chuyển định dạng video camera sang MJPEG khá đơn giản với công cụ v4l2-ctl. Hãy đảm bảo sử dụng đúng mã định dạng 'MJPG' và kiểm tra kỹ các bước để đạt kết quả mong muốn. Bạn có thể lưu tài liệu này thành file camera_mjpeg_setup.md để tham khảo sau.
Hy vọng tài liệu này hữu ích cho bạn!