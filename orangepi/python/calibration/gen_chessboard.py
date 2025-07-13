# generate_chessboard_A4_split_v2.py
import numpy as np
import cv2
from PIL import Image

# --- CẤU HÌNH BÀN CỜ TỐI ƯU ---
SQUARE_SIZE_MM = 25       # Kích thước mỗi ô vuông: 25mm
COLS = 7                  # 7 cột (6 góc trong)
ROWS = 20                 # 20 hàng (19 góc trong)
DPI = 300                 # Độ phân giải in

# --- TÍNH TOÁN KÍCH THƯỚC ---
MM_TO_INCH = 1 / 25.4
square_size_px = int(SQUARE_SIZE_MM * MM_TO_INCH * DPI)
border_px = square_size_px

# Kích thước trang A4 dọc (tính bằng pixel)
A4_H_PX = int(297 * MM_TO_INCH * DPI)
A4_W_PX = int(210 * MM_TO_INCH * DPI)

# 1. TẠO RA MỘT HỌA TIẾT BÀN CỜ LỚN (KHÔNG CÓ VIỀN)
board_h_px = ROWS * square_size_px
board_w_px = COLS * square_size_px
board_pattern = np.zeros((board_h_px, board_w_px), dtype=np.uint8)

for y in range(ROWS):
    for x in range(COLS):
        if (x + y) % 2 == 0:
            top_left = (x * square_size_px, y * square_size_px)
            bottom_right = ((x + 1) * square_size_px, (y + 1) * square_size_px)
            cv2.rectangle(board_pattern, top_left, bottom_right, 255, -1)

# 2. XÂY DỰNG TỪNG TRANG A4 RIÊNG BIỆT

# --- TẠO TRANG 1 (PHẦN TRÊN) ---
# Tạo một trang A4 trắng
page1_canvas = Image.new("L", (A4_W_PX, A4_H_PX), 255)
# Lấy phần trên của họa tiết bàn cờ
# Tính số hàng tối đa có thể vừa trên một trang, chừa lại viền trên và dưới
rows_for_page1 = (A4_H_PX - 2 * border_px) // square_size_px
height_for_page1 = rows_for_page1 * square_size_px
part1_np = board_pattern[:height_for_page1, :]
# Chuyển sang ảnh Pillow
part1_pil = Image.fromarray(part1_np)
# Dán vào giữa trang A4
offset_x = (A4_W_PX - part1_pil.width) // 2
offset_y = border_px # Canh lề trên
page1_canvas.paste(part1_pil, (offset_x, offset_y))


# --- TẠO TRANG 2 (PHẦN DƯỚI) ---
# Tạo một trang A4 trắng khác
page2_canvas = Image.new("L", (A4_W_PX, A4_H_PX), 255)
# Lấy phần dưới của họa tiết, bao gồm 1 hàng chồng lấp để dễ dán
start_row_for_part2 = rows_for_page1 - 1
start_pixel_for_part2 = start_row_for_part2 * square_size_px
part2_np = board_pattern[start_pixel_for_part2:, :]
# Chuyển sang ảnh Pillow
part2_pil = Image.fromarray(part2_np)
# Dán vào giữa trang A4
offset_x = (A4_W_PX - part2_pil.width) // 2
offset_y = border_px # Canh lề trên
page2_canvas.paste(part2_pil, (offset_x, offset_y))


# 3. LƯU 2 TRANG THÀNH 2 FILE PDF
try:
    output_pdf_1 = "chessboard_A4_part_1_top.pdf"
    output_pdf_2 = "chessboard_A4_part_2_bottom.pdf"
    
    page1_canvas.save(output_pdf_1, "PDF", resolution=DPI)
    page2_canvas.save(output_pdf_2, "PDF", resolution=DPI)
    
    print("✅ Đã tạo lại thành công 2 file PDF để in và ghép dọc:")
    print(f"   - File 1: {output_pdf_1} (Phần trên)")
    print(f"   - File 2: {output_pdf_2} (Phần dưới)")
    print("\nLưu ý: Tờ 'part_2' có một hàng ô vuông đầu tiên trùng với hàng cuối của tờ 'part_1' để bạn dễ dàng căn chỉnh và dán chồng lên.")

except Exception as e:
    print(f"\n❌ Đã xảy ra lỗi: {e}")
    print("Hãy chắc chắn bạn đã cài đặt thư viện Pillow: pip install Pillow")