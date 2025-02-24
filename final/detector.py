import cv2
import math
import time
import os
import numpy as np
from collections import deque
from datetime import datetime
import requests
import torch
from ultralytics import YOLO
import mediapipe as mp
from config import MODEL_PATH, SAVE_DIR
print(MODEL_PATH)

class PhoneDetector:
    def __init__(self):
        # Models
        self.phone_model = YOLO(MODEL_PATH).cuda() if torch.cuda.is_available() else YOLO(MODEL_PATH)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # State tracking
        self.history = deque(maxlen=30)  # 1 second @ 30 FPS
        self.alert_cooldown = 5

    def _get_landmarks_array(self, landmarks, frame_shape):
        """Convert normalized landmarks to pixel coordinates"""
        return np.array([(lm.x * frame_shape[1], lm.y * frame_shape[0]) 
                        for lm in landmarks], dtype=np.float32)

    def _analyze_hands(self, hand_landmarks, frame_shape):
        """Detect typing-specific hand configurations"""
        typing_signals = []
        for hand in hand_landmarks:
            points = self._get_landmarks_array(hand.landmark, frame_shape)
            
            # Finger-to-palm distances
            fingertips = points[[8, 12, 16, 20]]  # Tip landmarks
            palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
            
            distances = [np.linalg.norm(ft - palm_center) for ft in fingertips]
            avg_distance = np.mean(distances)
            
            # Typing pattern: fingers close to palm with slight spread
            typing_signals.append((
                avg_distance < 0.15 * frame_shape[1],  # Distance threshold
                np.std(distances) > 0.05 * frame_shape[1]  # Spread threshold
            ))
        
        return typing_signals

    def _calculate_confidence(self, signals):
        """Weighted confidence scoring"""
        weights = {
            'phone_detected': 0.5,  # Increased weight
            'hand_typing': 0.4,     # Increased weight
            'hand_proximity': 0.1
        }
        
        score = 0
        score += signals['phone_detected'] * weights['phone_detected']
        score += any(signals['hand_typing']) * weights['hand_typing']
        score += (signals['hand_distance'] < 0.2) * weights['hand_proximity']
        
        return min(max(score * 1.2, 0), 1)  # Scale to 0-1 range

    def process_frame(self, frame, last_alert_time):
        current_time = time.time()
        if current_time - last_alert_time < self.alert_cooldown:
            return frame, last_alert_time

        # Convert frame once for all models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Parallel inference
        phone_results = self.phone_model.predict(rgb_frame, verbose=False)
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Signal extraction
        signals = {
            'phone_detected': len(phone_results[0].boxes) > 0,
            'hand_typing': [],
            'hand_distance': 1.0
        }

        # Hand analysis
        if hand_results.multi_hand_landmarks:
            signals['hand_typing'] = self._analyze_hands(hand_results.multi_hand_landmarks, (h, w))
            
        # Pose analysis for hand proximity
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_wrist = np.array([landmarks[15].x * w, landmarks[15].y * h])
            right_wrist = np.array([landmarks[16].x * w, landmarks[16].y * h])
            signals['hand_distance'] = np.linalg.norm(left_wrist - right_wrist) / w

        # Temporal analysis
        self.history.append(signals)
        
        # Calculate moving confidence
        confidence = np.mean([self._calculate_confidence(s) for s in list(self.history)[-10:]])
        
        # Trigger condition (slightly increased threshold)
        if confidence > 0.88:  # 88% threshold
            # Visual feedback
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save and alert
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(SAVE_DIR, f"alert_{timestamp}.jpg"), frame)
            try:
                requests.post("http://localhost:5000/alert", json={"image_path": f"{SAVE_DIR}/alert_{timestamp}.jpg"}, timeout=1)
            except Exception as e:
                pass
            last_alert_time = current_time

        return frame, last_alert_time