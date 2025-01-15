
# Security System

For rane, installation process is given below



## Installation

Download the zip file or clone the repo using the command

```bash
git clone https://github.com/aijurist/rane_email-detection.git
```
\
Navigate to the source directory using the command 
```bash
cd rane_email-detection
```

Once cloned / extracted, create a python virtual environment by using the following commands, replace ```<environment-name>``` with some other name such as rane
```bash
python -m venv <environment-name>
```
Initialize the virtual environment by using the command
```bash
./<environment-name>/Scripts/activate
```
\
Now install all the required packages using the command 
```bash
pip install -r requirements.txt
```
\
Set the environment variables in the file ```.env.sample``` and rename the file to .env
```bash
# Path to the audio file for alert notifications
AUDIO_FILE=alert.mp3

# Directory to save captured images
SAVE_DIR=img

# Path to the YOLO model file
MODEL_PATH=yolov8n.pt

# RTSP URL for the camera feed
RTSP_URL=rtsp://<username>:<password>@<camera-ip>:<port>/Streaming/Channels/2

# Email configuration
SEND_FROM=your-email@gmail.com
SEND_TO=recipient1@example.com,recipient2@example.com
APP_PASSWORD=your-app-specific-password

# Alert delay in seconds
ALERT_DELAY=5
```

Now run the camera.py script in the ```src``` folder 
