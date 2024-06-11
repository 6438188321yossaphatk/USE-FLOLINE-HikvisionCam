import cv2
from datetime import datetime
from ultralytics import YOLO, solutions
import requests
from pymodbus.client import ModbusTcpClient
import os

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the webcam feed
source = "rtsp://admin:sec0mmth@192.168.254.14:554/Streaming/Channels/101"  # Add your stream sources here
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Define the Modbus server address and port
modbus_server_address = '192.168.99.183'
modbus_server_port = 502  # Default Modbus TCP port

# Define the server endpoint for HTTP frame upload
server_url = "http://192.168.40.185:5432/upload_frame"

# Get video properties
original_w, original_h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Reduce frame size for processing
processing_scale = 0.3
w = int(original_w * processing_scale)
h = int(original_h * processing_scale)
w1 = int(w * 0.2)
w2 = int(w * 0.8)
h_half = int(h / 2)
region_points = [(w1, h_half), (w2, h_half)]

# Initialize Object Counter
counter = solutions.ObjectCounter(
    classes_names=model.names,
    reg_pts=region_points,
    view_img=False,  # Set to True to display the frames while processing
    draw_tracks=False,
)

# Process the video frames
frame_skip = 20  # Process every 2nd frame
frame_count = 0

# Define the directory to save recordings
recordings_dir = 'recordings'
os.makedirs(recordings_dir, exist_ok=True)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
now = datetime.now()
video_filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
video_path = os.path.join(recordings_dir, video_filename)
out = cv2.VideoWriter(video_path, fourcc, 20, (w, h))  # Set the playback speed to 30 fps

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    im0_resized = cv2.resize(im0, (w, h))
    tracks = model.track(im0_resized, persist=True, show=False)

    # Start counting
    im0_resized = counter.start_counting(im0_resized, tracks)

    # Extract the counts directly from the counter object
    num_entered = counter.in_counts
    num_left = counter.out_counts

    # Write the frame to the video
    out.write(im0_resized)

    # Send frame to HTTP server
    _, buffer = cv2.imencode('.jpg', im0_resized)
    frame_bytes = buffer.tobytes()
    headers = {'Content-Type': 'application/octet-stream'}
    try:
        response = requests.post(server_url, data=frame_bytes, headers=headers)
        if response.status_code != 200:
            print(f"Failed to send frame to server, status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending frame to server: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Release the video writer
out.release()

print(f"Video saved: {video_path}")
