import cv2
import time
import threading
from config import RTSP_URL, ALERT_DELAY
from detector import PhoneDetector

class FrameReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                print("Frame read error, reconnecting...")
                time.sleep(1)
                self.reconnect()
                continue
            with self.lock:
                self.frame = frame

    def reconnect(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_latest_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def main():
    detector = PhoneDetector()
    reader = FrameReader(RTSP_URL)

    # time.sleep(2)

    if not reader.cap.isOpened():
        print("Error: Could not connect to camera feed")
        return

    last_alert_time = 0
    frame_counter = 0

    try:
        while True:
            start_time = time.time()
            frame = reader.get_latest_frame()

            if frame is None:
                print("No frame available, waiting...")
                time.sleep(0.1)
                continue

            # Preprocessing
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.resize(frame, (640, 480))
            frame_counter += 1

            # Process every 5th frame
            if frame_counter % 5 == 0:
                processed_frame, last_alert_time = detector.process_frame(
                    frame, last_alert_time, ALERT_DELAY
                )
                display_frame = processed_frame
            else:
                display_frame = frame

            # Display
            cv2.imshow("Camera Feed", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Reset counter after processing
            if frame_counter % 5 == 0:
                frame_counter = 0

    finally:
        reader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()