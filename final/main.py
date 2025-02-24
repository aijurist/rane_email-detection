import cv2
import time
import threading
from config import RTSP_URL, SAVE_DIR
from detector import PhoneDetector

class FrameReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._connect()

    def _connect(self):
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return self.cap.isOpened()

    def _update_frames(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(1)
                self.cap.release()
                if not self._connect():
                    continue
                else:
                    print("Reconnected to stream")
                continue
            
            with self.lock:
                # Preprocess immediately after capture
                self.frame = cv2.resize(frame, (640, 480))
                self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_frames, daemon=True)
            self.thread.start()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.thread:
            self.thread.join(timeout=2)

def main():
    detector = PhoneDetector()
    reader = FrameReader(RTSP_URL)
    reader.start()

    last_alert_time = 0
    fps_last = time.time()
    fps_counter = 0
    fps = 0

    try:
        while True:
            start_time = time.time()
            frame = reader.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # Process every frame
            processed_frame, last_alert_time = detector.process_frame(frame, last_alert_time)

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_last >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_last = time.time()

            # Add performance overlay
            cv2.putText(processed_frame, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Last alert: {time.time() - last_alert_time:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display output
            cv2.imshow("Surveillance Monitor", processed_frame)

            # Exit handling
            if cv2.waitKey(1) in [ord('q'), 27]:  # Q or ESC
                break

            # Maintain processing rate
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # ~30 FPS
                time.sleep(0.033 - elapsed)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        reader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()