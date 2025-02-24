from flask import Flask, request, jsonify
from datetime import datetime
import pyttsx3
from config import SEND_FROM, SEND_TO, APP_PASSWORD
from send_mail import send_mail

app = Flask(__name__)

@app.route('/alert', methods=['POST'])
def trigger_alert():
    data = request.get_json()
    image_path = data.get("image_path")
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400

    # Create the alert text with current timestamp
    text = f"Phone usage detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Send email alert
    try:
        send_mail(
            send_from=SEND_FROM,
            send_to=SEND_TO,
            subject="Phone Usage Alert",
            text=text,
            file_path=image_path,
            app_password=APP_PASSWORD
        )
    except Exception as e:
        return jsonify({"error": f"Failed to send email: {e}"}), 500

    # Play TTS alert
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say("Cell phone usage is detected.")
        engine.runAndWait()
    except Exception as e:
        return jsonify({"error": f"Failed to play TTS alert: {e}"}), 500

    return jsonify({"status": "alert triggered"}), 200

if __name__ == "__main__":
    # Listen on all interfaces, port 5000 (adjust if needed)
    app.run(host="0.0.0.0", port=5000)
