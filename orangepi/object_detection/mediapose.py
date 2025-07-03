import cv2
import mediapipe as mp
from mediapipe import solutions
import matplotlib.pyplot as plt 

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

# Read the image
image = cv2.imread("image.webp")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform Pose Detection
results = pose.process(rgb_image)

# Draw pose landmarks if they were detected
if results.pose_landmarks:
    # Draw landmarks and connections on the image
    annotated_image = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert to RGB for displaying with matplotlib
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()
else:
    print("No pose detected in the image.")