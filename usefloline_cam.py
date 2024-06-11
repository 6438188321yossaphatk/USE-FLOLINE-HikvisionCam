import cv2
from datetime import datetime
from ultralytics import YOLO, solutions
import requests
from pymodbus.client import ModbusTcpClient
import os
from threading import Thread, Lock

# Global variables
frame_queue = []
frame_queue_lock = Lock()
terminate = False

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

# Define the directory to save recordings
recordings_dir = 'recordings'
os.makedirs(recordings_dir, exist_ok=True)

# Function to read frames from the video stream
def read_frames():
    global frame_queue, terminate
    while cap.isOpened() and not terminate:
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        with frame_queue_lock:
            frame_queue.append(frame)

# Function to preprocess frames
def preprocess_frame(frame):
    return cv2.resize(frame, (w, h))

# Function to calculate region points based on frame size and processing scale
def calculate_region_points():
    w1 = int(w * 0.2)
    w2 = int(w * 0.8)
    h_half = int(h / 2)
    return [(w1, h_half), (w2, h_half)]

# Initialize Object Counter
counter = solutions.ObjectCounter(
    classes_names=model.names,
    reg_pts=calculate_region_points(),
    view_img=True,  # Set to True to display the frames while processing
    draw_tracks=False,
)

# Function to process frames (object detection)
def process_frames():
    global frame_queue, terminate
    while cap.isOpened() or len(frame_queue) > 0:
        if terminate and len(frame_queue) == 0:
            break
        with frame_queue_lock:
            if len(frame_queue) == 0:
                continue
            frame = frame_queue.pop(0)
        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)
        # Perform object detection
        tracks = model.track(preprocessed_frame, persist=True, show=False)
        # Count objects
        processed_frame = counter.start_counting(preprocessed_frame, tracks)
        # Send frame to HTTP server
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            response = requests.post(server_url, data=frame_bytes, headers=headers)
            if response.status_code != 200:
                print(f"Failed to send frame to server, status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending frame to server: {e}")

# Function to write frames to video file
def write_frames():
    global frame_queue, terminate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    now = datetime.now()
    video_filename = f"output_{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
    video_path = os.path.join(recordings_dir, video_filename)
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
    while cap.isOpened() or len(frame_queue) > 0:
        if terminate and len(frame_queue) == 0:
            break
        with frame_queue_lock:
            if len(frame_queue) == 0:
                continue
            frame = frame_queue.pop(0)
        out.write(frame)
    out.release()
    print(f"Video saved: {video_path}")

# Create and start threads
read_thread = Thread(target=read_frames)
process_thread = Thread(target=process_frames)
write_thread = Thread(target=write_frames)
read_thread.start()
process_thread.start()
write_thread.start()

# Main loop to check for 'q' key press to terminate the program
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        terminate = True
        break

# Wait for threads to finish
read_thread.join()
process_thread.join()
write_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()
