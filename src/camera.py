from ultralytics import YOLO
import cv2
import threading
import os
import sys
from datetime import datetime
import pygame
import time
from dotenv import load_dotenv
# import logging
# import mylib
# logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants from environment variables
AUDIO_FILE = os.getenv("AUDIO_FILE", "alert.mp3")
SAVE_DIR = os.getenv("SAVE_DIR", "img")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
RTSP_URL = os.getenv("RTSP_URL")
SEND_FROM = os.getenv("SEND_FROM")
SEND_TO = os.getenv("SEND_TO").split(",")
APP_PASSWORD = os.getenv("APP_PASSWORD")
ALERT_DELAY = int(os.getenv("ALERT_DELAY", "5"))

os.makedirs(SAVE_DIR, exist_ok=True)
pygame.mixer.init()

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)
from send_mail import send_mail

last_alert_time = 0
model = YOLO(MODEL_PATH)

# Class names for the YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


# Function to send email alerts
def send_alert_email(image_path):
    text = f"Phone detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        send_mail(
            send_from='rmlp3cctv@gmail.com',
            send_to=['s.mohamedazardeen@ranegroup.com', 'shanthosh811@gmail.com', 'ghadhatharan.v@rajalakshmi.edu.in', 'm.sanjay@ranegroup.com'],
            subject="Phone Detection Alert",
            text=text,
            file_path=image_path,
            app_password='nhmy bsbw twrk foes'
        )
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Function to process frames
def process_frame(frame, last_alert_time, alert_delay):
    global model
    
    current_time = time.time()
    if (current_time - last_alert_time) < alert_delay:
        cv2.imshow("Camera Feed", frame)
        return last_alert_time
    
    results = model(frame, stream=True, conf=0.5, classes=[67])
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Phone: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(SAVE_DIR, f"phone_detected_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            
            
            def handle_alerts():
                threading.Thread(target=send_alert_email, args=(image_path,), daemon=True).start()
                pygame.mixer.music.load(AUDIO_FILE)
                pygame.mixer.music.play()
            
            threading.Thread(target=handle_alerts, daemon=True).start()
            return current_time
    
    cv2.imshow("Camera Feed", frame)
    return last_alert_time

# Main function
# def main():
#     global last_alert_time
#     cap = cv2.VideoCapture(RTSP_URL)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     if not cap.isOpened():
#        print("Error: Could not connect to camera feed")
#        return
#     cv2.setNumThreads(4)

#    try:
#        while True:
#            success, frame = cap.read()
#            if not success:
#                print("Error: Failed to grab frame from camera feed")
#                break
#            last_alert_time = process_frame(frame, last_alert_time, ALERT_DELAY)
#            time.sleep(0.03)

#    finally:
#        cap.release()
#        print("Camera feed closed.")

# comment the above main function and uncomment the below main function to run the code with GUI

def main():
     global last_alert_time
     cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
     cap.set(cv2.CAP_PROP_FPS,30)
     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
     cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

     if not cap.isOpened():
         print("Error: Could not connect to camera feed")
         return
    #  cv2.setNumThreads(4)

     try:
         while True:
             success, frame = cap.read()
             if not success:
                 print("Error: Failed to grab frame from camera feed")
                 time.sleep(0.1)
                 continue
                #  break
             # frame resize
             frame = cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
             last_alert_time = process_frame(frame, last_alert_time, ALERT_DELAY)
             cv2.imshow("Camera Feed", frame)
        

             # Exit the loop if 'q' is pressed
             if cv2.waitKey(1) & 0xFF == ord('q'):
                 print("Exiting...")
                 break

            #  time.sleep(0.03)

     finally:
         cap.release()
         cv2.destroyAllWindows()
         print("Camera feed closed.")

if __name__ == "__main__":
    main()
