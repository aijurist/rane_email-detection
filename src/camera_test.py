import cv2
import threading
import os
import sys
from datetime import datetime
import pygame
import time
import math
from dotenv import load_dotenv
from ultralytics import YOLO
import mediapipe as mp

# Set up directories and environment
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)
from send_mail import send_mail

load_dotenv()

# Constants from environment variables
AUDIO_FILE = os.getenv("AUDIO_FILE", "alert.mp3")
SAVE_DIR = os.getenv("SAVE_DIR", "img")
RTSP_URL = os.getenv("RTSP_URL")
ALERT_DELAY = int(os.getenv("ALERT_DELAY", "5"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

os.makedirs(SAVE_DIR, exist_ok=True)
pygame.mixer.init()

last_alert_time = 0

# Initialize YOLO model for phone detection (class 67 should correspond to phone)
yolo_model = YOLO(MODEL_PATH)  # Make sure your model is trained/detects phone class reliably

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def send_alert_email(image_path):
    text = f"Phone usage detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        send_mail(
            send_from=os.getenv("SEND_FROM"),
            send_to=os.getenv("SEND_TO").split(","),
            subject="Phone Usage Alert",
            text=text,
            file_path=image_path,
            app_password=os.getenv("APP_PASSWORD")
        )
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def classify_phone_usage(landmarks, phone_center, frame_width, frame_height):
    """
    Given pose landmarks and phone center, this function checks:
      - Is the phone near either wrist? If not, we assume it’s not being held.
      - If held, is it near an ear (suggesting talking) or near the face center (suggesting texting)?
      - Otherwise, we classify it as simply "holding phone".
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
        # If any required landmark is missing, return no classification
        return None

    def distance(pt1, pt2):
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
    
    # Check proximity to wrists (ensure phone is actually being held)
    wrist_threshold = 0.1 * frame_width  # adjust threshold as needed
    dist_left_wrist = distance(phone_center, left_wrist)
    dist_right_wrist = distance(phone_center, right_wrist)
    if dist_left_wrist > wrist_threshold and dist_right_wrist > wrist_threshold:
        return None  # Not held close to a hand

    # Determine if phone is near ear (suggesting a call/talking scenario)
    ear_threshold = 0.1 * frame_width  # adjust threshold
    dist_left_ear = distance(phone_center, left_ear)
    dist_right_ear = distance(phone_center, right_ear)
    if dist_left_ear < ear_threshold or dist_right_ear < ear_threshold:
        return "talking while walking"

    # Determine if phone is near face center (texting scenario)
    face_threshold = 0.15 * frame_width  # adjust threshold
    if distance(phone_center, nose) < face_threshold:
        return "texting"

    # Otherwise, classify as simply holding the phone
    return "holding phone"

def process_frame(frame, last_alert_time, alert_delay):
    current_time = time.time()
    frame_height, frame_width = frame.shape[:2]

    # Enforce alert delay
    if (current_time - last_alert_time) < alert_delay:
        cv2.imshow("Camera Feed", frame)
        return last_alert_time

    # First, use YOLO to detect phones
    yolo_results = yolo_model(frame, stream=True, conf=0.2, classes=[67])
    phone_detected = False
    phone_bbox = None
    phone_conf = 0

    for r in yolo_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            phone_bbox = [int(coord) for coord in box.xyxy[0].tolist()]
            phone_conf = conf
            phone_detected = True
            break
        if phone_detected:
            break

    if not phone_detected:
        cv2.imshow("Camera Feed", frame)
        return last_alert_time

    # Calculate phone center from YOLO bbox
    x1, y1, x2, y2 = phone_bbox
    phone_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Next, run MediaPipe Pose to get key body landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    scenario = None
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        scenario = classify_phone_usage(landmarks, phone_center, frame_width, frame_height)

    # If a valid scenario is detected, annotate and alert
    if scenario:
        # Draw the phone bounding box and phone center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, phone_center, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{scenario} ({phone_conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame and trigger alerts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(SAVE_DIR, f"phone_usage_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

        def handle_alerts():
            threading.Thread(target=send_alert_email, args=(image_path,), daemon=True).start()
            pygame.mixer.music.load(AUDIO_FILE)
            pygame.mixer.music.play()
        
        threading.Thread(target=handle_alerts, daemon=True).start()
        last_alert_time = current_time

    cv2.imshow("Camera Feed", frame)
    return last_alert_time

def main():
    global last_alert_time
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("Error: Could not connect to camera feed")
        return

    frame_counter = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Frame read error, reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                continue

            # Rotate the frame 180° if it appears upside down
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.resize(frame, (640, 480))
            frame_counter += 1

            # Process every 5th frame
            if frame_counter % 5 == 0:
                last_alert_time = process_frame(frame, last_alert_time, ALERT_DELAY)
            else:
                cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
