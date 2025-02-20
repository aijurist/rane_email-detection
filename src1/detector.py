# detector.py
import cv2
import math
import time
import threading
import os
from datetime import datetime
import torch
from ultralytics import YOLO
import mediapipe as mp
from config import MODEL_PATH, SAVE_DIR
from alerts import trigger_alerts

class PhoneDetector:
    def __init__(self):
        # Initialize YOLO model on GPU if available
        self.yolo_model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            self.yolo_model = self.yolo_model.cuda()
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    @staticmethod
    def classify_phone_usage(landmarks, phone_center, frame_width, frame_height):
        """
        Classify phone usage based on pose landmarks and phone location.
        Returns:
          - "talking while walking" if near ear,
          - "texting" if near face,
          - "holding phone" if held,
          - or None if the phone is not held.
        """
        def to_pixel(landmark):
            return int(landmark.x * frame_width), int(landmark.y * frame_height)
        
        try:
            nose = to_pixel(landmarks[0])
            left_wrist = to_pixel(landmarks[15])
            right_wrist = to_pixel(landmarks[16])
            left_ear = to_pixel(landmarks[7])
            right_ear = to_pixel(landmarks[8])
        except IndexError:
            return None

        def distance(pt1, pt2):
            return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        
        # Check if phone is near either wrist
        wrist_threshold = 0.1 * frame_width
        if distance(phone_center, left_wrist) > wrist_threshold and distance(phone_center, right_wrist) > wrist_threshold:
            return None
        
        # Check if phone is near ear (suggesting talking)
        ear_threshold = 0.1 * frame_width
        if distance(phone_center, left_ear) < ear_threshold or distance(phone_center, right_ear) < ear_threshold:
            return "talking while walking"
        
        # Check if phone is near face (suggesting texting)
        face_threshold = 0.15 * frame_width
        if distance(phone_center, nose) < face_threshold:
            return "texting"
        
        return "holding phone"

    def process_frame(self, frame, last_alert_time, alert_delay):
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]

        # Skip processing during cooldown period
        if (current_time - last_alert_time) < alert_delay:
            return frame, last_alert_time

        # Run YOLO inference with optimized parameters
        yolo_results = self.yolo_model.predict(
            frame,
            conf=0.2,
            classes=[67],
            imgsz=320,
            agnostic_nms=True,
            max_det=1
        )

        phone_detected = False
        phone_bbox = None
        phone_conf = 0

        # Process YOLO results
        for r in yolo_results:
            if r.boxes:
                box = r.boxes[0]  # Take first detection
                phone_bbox = [int(coord) for coord in box.xyxy[0].tolist()]
                phone_conf = box.conf.item()
                phone_detected = True
                break

        if not phone_detected:
            return frame, last_alert_time

        x1, y1, x2, y2 = phone_bbox
        phone_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        scenario = None
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            scenario = self.classify_phone_usage(landmarks, phone_center, frame_width, frame_height)

        if scenario:
            # Annotate frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, phone_center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{scenario} ({phone_conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save frame in a thread
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(SAVE_DIR, f"phone_usage_{timestamp}.jpg")
            threading.Thread(target=cv2.imwrite, args=(image_path, frame), daemon=True).start()

            # Trigger alerts in a separate thread
            threading.Thread(target=trigger_alerts, args=(image_path,), daemon=True).start()
            last_alert_time = current_time

        return frame, last_alert_time