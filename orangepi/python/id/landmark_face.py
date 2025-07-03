import cv2
import numpy as np
import logging
import mediapipe as mp

class LandmarkFace: 
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
        self.connections = {
            "RIGHT_IRIS": mp.solutions.face_mesh_connections.FACEMESH_RIGHT_IRIS,
            "TESSELATION": mp.solutions.face_mesh_connections.FACEMESH_TESSELATION,
            "RIGHT_EYE": mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYE,
            "RIGHT_EYEBROW": mp.solutions.face_mesh_connections.FACEMESH_RIGHT_EYEBROW,
            "NOSE": mp.solutions.face_mesh_connections.FACEMESH_NOSE,
            "LIPS": mp.solutions.face_mesh_connections.FACEMESH_LIPS,
            "LEFT_IRIS": mp.solutions.face_mesh_connections.FACEMESH_LEFT_IRIS,
            "LEFT_EYEBROW": mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYEBROW,
            "LEFT_EYE": mp.solutions.face_mesh_connections.FACEMESH_LEFT_EYE,
            "IRISES": mp.solutions.face_mesh_connections.FACEMESH_IRISES,
            "FACE_OVAL": mp.solutions.face_mesh_connections.FACEMESH_FACE_OVAL,
            "CONTOURS": mp.solutions.face_mesh_connections.FACEMESH_CONTOURS,
        }

    def detect_landmarks(self, frame):
        if not isinstance(frame, np.ndarray):
            logging.error("Invalid frame: not a numpy array")
            return None, None, None

        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                logging.error("No face landmarks detected.")
                return None, None, None

            face_landmarks_out = results.multi_face_landmarks[0]
            image_height, image_width, _ = frame.shape
            
            # Convert landmarks to numpy array with correct shape
            landmarks = np.array([(lm.x * image_width, lm.y * image_height) 
                                for lm in face_landmarks_out.landmark])
            
            if landmarks.shape[0] == 0:
                logging.error("No landmarks detected")
                return None, None, None

            # Calculate bounding box
            x_min = int(np.min(landmarks[:, 0]))
            y_min = int(np.min(landmarks[:, 1]))
            x_max = int(np.max(landmarks[:, 0]))
            y_max = int(np.max(landmarks[:, 1]))
            
            # Add padding to face box
            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            width = min(image_width - x_min, x_max - x_min + 2*padding)
            height = min(image_height - y_min, y_max - y_min + 2*padding)
            
            face_box = (x_min, y_min, width, height)

            return image_rgb, landmarks, face_box

        except Exception as e:
            logging.error(f"Error in landmark detection: {str(e)}")
            return None, None, None

    def close(self):
        self.face_mesh.close()

