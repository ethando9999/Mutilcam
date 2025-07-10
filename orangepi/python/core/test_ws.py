# file: test_ws_array.py
import websocket # Cần cài đặt: pip install websocket-client
import json
import time

# --- THÔNG SỐ TEST ---
WEBSOCKET_URL = "ws://192.168.1.168:8080/api/ws/camera"
TABLE_ID = 1
# Mảng dữ liệu chiều cao mẫu (đơn vị cm) để gửi đi
SAMPLE_HEIGHTS_CM = [156.5, 153.2, 158.9, 154.2] 
 
def run_test():
    """
    Kết nối và gửi một payload chứa mảng chiều cao.
    """
    # Định dạng payload mới
    payload = {
        "table_id": TABLE_ID, 
        "heights_cm": SAMPLE_HEIGHTS_CM 
        # Gửi một key mới là 'heights_cm' chứa một mảng
    }
    
    ws = None
    try:
        print(f"Đang kết nối tới {WEBSOCKET_URL}...") 
        ws = websocket.create_connection(WEBSOCKET_URL, timeout=10)
        print(">>> Kết nối thành công!") 

        # Gửi dữ liệu
        json_payload = json.dumps(payload)
        print(f"--> Đang gửi: {json_payload}")
        ws.send(json_payload)

        # Đợi và nhận phản hồi từ server
        print("Đã gửi. Đang chờ phản hồi...")
        result = ws.recv() 
        print(f"<-- Phản hồi từ Server: {result}")

    except Exception as e:
        print(f"!!! Lỗi: {e}")
    finally:
        if ws:
            ws.close()
            print("Đã đóng kết nối.")

if __name__ == "__main__":
    run_test()
