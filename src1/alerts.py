import threading
from datetime import datetime
from config import SEND_FROM, SEND_TO, APP_PASSWORD
from send_mail import send_mail
import pyttsx3

def send_alert_email(image_path):
    """Send an email alert with an attached image."""
    text = f"Phone usage detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        send_mail(
            send_from=SEND_FROM,
            send_to=SEND_TO,
            subject="Phone Usage Alert",
            text=text,
            file_path=image_path,
            app_password=APP_PASSWORD
        )
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def play_tts_alert():
    """Use pyttsx3 to speak an alert message."""
    try:
        engine = pyttsx3.init()
        # Optionally adjust the speech rate:
        rate = engine.getProperty('rate')
        engine.setProperty('rate', 160)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        message = "Cell phone usage is detected."
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Failed to play TTS alert: {e}")

def trigger_alerts(image_path):
    threading.Thread(target=send_alert_email, args=(image_path,), daemon=True).start()
    play_tts_alert()
    # send_alert_email(image_path)

# if __name__ == "__main__":
#     trigger_alerts(r"img\phone_usage_20250219_052502.jpg")
