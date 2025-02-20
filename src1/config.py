# config.py
import os
from dotenv import load_dotenv

load_dotenv()

AUDIO_FILE = os.getenv("AUDIO_FILE", "alert.mp3")
SAVE_DIR = os.getenv("SAVE_DIR", "img")
RTSP_URL = os.getenv("RTSP_URL")
ALERT_DELAY = int(os.getenv("ALERT_DELAY", "5"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov10n.pt")
SEND_FROM = os.getenv("SEND_FROM")
SEND_TO = os.getenv("SEND_TO").split(",") if os.getenv("SEND_TO") else []
APP_PASSWORD = os.getenv("APP_PASSWORD")
