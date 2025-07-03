import socket
import os
import cv2
import numpy as np
import threading
import time
from datetime import datetime

# Configurations
SOCKET_PATH = "/mnt/ramdisk/unix_socket"
OUTPUT_DIR = "received_frames"
MAX_QUEUE_SIZE = 10

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Frame queue and lock for thread-safe operations
frame_queue = []
queue_lock = threading.Lock()

# Global variable to track frames processed and time for FPS calculation
last_time = time.time()
frames_processed = 0

def save_frame(frame_data):
    """Save the frame as an image in the output directory."""
    global last_time, frames_processed
    try:
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Error: Unable to decode frame.")
            return

        # Format timestamp as datetime string
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        file_name = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.jpg")

        # Measure FPS just before saving the frame
        current_time = time.time()
        if current_time - last_time >= 1:
            fps = frames_processed / (current_time - last_time)
            print(f"Frames processed: {frames_processed}, FPS: {fps:.2f}")
            frames_processed = 0
            last_time = current_time
        frames_processed += 1

        # Save the frame
        cv2.imwrite(file_name, frame)
        print(f"Frame saved as {file_name}")
    except Exception as e:
        print(f"Error saving frame: {e}")


def process_frames():
    """Process frames from the queue."""
    while True:
        queue_lock.acquire()
        if frame_queue:
            frame_data = frame_queue.pop(0)
            queue_lock.release()

            # Process the frame (e.g., save it)
            save_frame(frame_data)
        else:
            queue_lock.release()
            time.sleep(0.1)  # Avoid busy-waiting


def handle_unix_client(conn):
    """Handle Unix socket client connection."""
    try:
        frame_data = b""
        while True:
            chunk = conn.recv(1024*1024)

            if not chunk:
                if frame_data:
                    # Process completed frame
                    queue_lock.acquire()
                    if len(frame_queue) >= MAX_QUEUE_SIZE:
                        print("Queue is full, dropping frame...")
                    else:
                        frame_queue.append(frame_data)
                    queue_lock.release()
                    frame_data = b""  # Reset buffer
                print("No frame data received. Closing connection.")
                break

            # Check for marker indicating end of frame
            if chunk.endswith(b"\x00"):
                frame_data += chunk.rstrip(b"\x00")

                # Add frame to queue
                queue_lock.acquire()
                if len(frame_queue) >= MAX_QUEUE_SIZE:
                    print("Queue is full, dropping frame...")
                else:
                    frame_queue.append(frame_data)
                queue_lock.release()

                frame_data = b""  # Reset buffer for the next frame
            else:
                frame_data += chunk  # Append chunk to the buffer
    except Exception as e:
        print(f"Error handling Unix client: {e}")
    finally:
        conn.close()


def start_unix_socket_server():
    """Start the Unix socket server."""
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(1)
    print("Unix Socket server is ready and listening.")

    while True:
        conn, _ = server_socket.accept()
        print("Connection from Unix client.")
        threading.Thread(target=handle_unix_client, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    # Start frame processing thread
    threading.Thread(target=process_frames, daemon=True).start()

    # Start Unix socket server
    start_unix_socket_server()