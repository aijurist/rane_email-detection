from onvif import ONVIFCamera
import subprocess

# Initialize camera connection (verify WSDL path)
cam = ONVIFCamera('192.168.1.9', 80, 'admin', 'NEAWKH', 
                 r'D:\rane\.venv\Lib\site-packages\wsdl')

# Create media service
media_service = cam.create_media_service()

# Get profiles with proper token extraction
profiles = media_service.GetProfiles()
profile_token = profiles[0].token  # Use .token instead of ._token

# Configure stream parameters
stream_setup = {
    'Stream': 'RTP-Unicast',
    'Transport': {
        'Protocol': 'RTSP',
        'Tunnel': None
    }
}

# Get stream URI with authentication embedding
request = media_service.create_type('GetStreamUri')
request.ProfileToken = profile_token
request.StreamSetup = stream_setup

stream_uri = media_service.GetStreamUri(request)
rtsp_url = stream_uri.Uri.replace('rtsp://', 'rtsp://admin:NEAWKH@')  # Add credentials

print(f"Audio Stream URI: {rtsp_url}")

# Add this after getting profiles
print("Camera capabilities:", media_service.GetServiceCapabilities())
print("Audio configurations:", media_service.GetAudioSources())

# FFmpeg command with proper audio conversion
subprocess.run([
    'ffmpeg',
    '-loglevel', 'debug',
    '-i', 'alert.mp3',
    '-vn',
    '-c:a', 'aac',
    '-ar', '8000',
    '-ac', '1',
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    '-payload_type', '8',
    'rtsp://admin:NEAWKH@192.168.1.9:554/Streaming/Channels/2'
], check=True)