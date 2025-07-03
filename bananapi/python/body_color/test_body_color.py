import cv2
from object_detection.yolo_pose import HumanDetection
from bananapi.python.body_color.pose_cluster import PoseClusterProcessor

# Khởi tạo đối tượng detect human
detector = HumanDetection()

# Khởi tạo đối tượng xử lý cụm màu
pose_processor = PoseClusterProcessor()

# Hàm xử lý kết quả detection
def process_detection(image):
    # Xác định các cảm xúc đã phát hiện trong ảnh
    keypoints_data, boxes_data = detector.run_detection(image)

    # Xử lý cụm màu
    body_color_data = []
    for keypoints in keypoints_data:
        edge_pixels = pose_processor.extract_edge_pixels(image, keypoints)
        body_color = pose_processor.get_body_color_signature(edge_pixels)

        body_color_data.append(body_color)
        
    # Trả về danh sách màu và thông tin cụm màu
    return keypoints_data, boxes_data, body_color_data

def main():
    image_file = "python/file.jpg"
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not read image {image_file}")
        return
    
    keypoints_data, boxes_data, body_color_data = process_detection(image)
    pose_processor.visualize_color_signatures(body_color_data)

if __name__ == "__main__":
    main()
