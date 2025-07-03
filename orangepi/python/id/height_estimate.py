class HeightEstimator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.face_landmark_detector = LandmarkFace()

    def estimate_height(self):
        annotated_image, face_landmarks_out, region_boxes = self.face_landmark_detector.detect_landmarks(self.image_path)

        if annotated_image is not None:
            estimate_head = head_height.EstimateHead(pixel_to_cm_ratio=0.12)

            distances = {
                ("RIGHT_EYE", "NOSE"): 9.0,
                ("LEFT_EYE", "NOSE"): 9.0,
                ("NOSE", "LIPS"): 8.5,
                ("RIGHT_EYE", "LEFT_EYE"): 6.5,
                ("NOSE", "RIGHT_EYEBROW"): 8.8,
                ("NOSE", "LEFT_EYEBROW"): 8.8,
                ("LIPS", "FACE_OVAL"): 8.8,
                ("RIGHT_EYEBROW", "RIGHT_EYE"): 8.3,
                ("LEFT_EYEBROW", "LEFT_EYE"): 8.3,
                ("LEFT_EYE", "LEFT_IRIS"): 84.5,
                ("RIGHT_EYE", "RIGHT_IRIS"): 84.5,
                ("RIGHT_EYE", "FACE_OVAL"): 6.5,
                ("LEFT_EYE", "FACE_OVAL"): 6.5,
                ("LEFT_EYEBROW", "FACE_OVAL"): 7.5,
                ("RIGHT_EYEBROW", "FACE_OVAL"): 7.5,
            }

            head_height = estimate_head.estimate_head_height(region_boxes, distances)

            if head_height:
                print(f"Estimated Head Height: {head_height:.2f} cm")
            else:
                print("Unable to estimate head height.")

            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Annotated Image")
            # plt.show()

            for region_name, bbox in region_boxes.items():
                x, y, w, h = bbox
                print(f"{region_name}: x={x}, y={y}, w={w}, h={h}")

            try:
                face_region = "FACE_OVAL"
                if face_region not in region_boxes:
                    logging.error(f"Face region '{face_region}' not found in region_boxes.")
                    return

                x, y, w, h = region_boxes[face_region]
                cropped_face = annotated_image[y:y + h, x:x + w]

                if cropped_face is None or cropped_face.size == 0:
                    logging.error("Failed to crop the face region from the image.")
                    return

                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                face_analyzer = AnalyzeFace()
                result = face_analyzer.analyze(cropped_face_rgb)

                if result:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    age = result.get('age')
                    dominant_gender = result.get('dominant_gender')
                    dominant_emotion = result.get('dominant_emotion')
                    dominant_race = result.get('dominant_race')
                    logging.info(f"[{timestamp}] Age: {age}, Gender: {dominant_gender}, Emotion: {dominant_emotion}, Race: {dominant_race}")

                    if head_height:
                        base_height = head_height * 7.7
                        estimator = height.EstimatePersonHeight(
                            base_height_cm=base_height,
                            gender=dominant_gender,
                            age=age,
                            race=dominant_race,
                            emotion=dominant_emotion
                        )
                        total_height = estimator.estimate_total_height()
                        logging.info(f"Estimated Total Height: {total_height:.2f} cm")
                        print(f"Estimated Total Height: {total_height:.2f} cm")

                else:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logging.info(f"[{timestamp}] No result returned from analysis.")

            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logging.error(f"[{timestamp}] Unexpected error: {e}")

        self.face_landmark_detector.close()