import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO, solutions
import requests
from pymodbus.client import ModbusTcpClient
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Global variables for camera and video writer
cap = None
out = None
thread_pool = None

# Function to handle termination signals
def signal_handler(sig, frame):
    print('Exiting the program...')
    if cap is not None:
        cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    if thread_pool is not None:
        thread_pool.shutdown(wait=True)
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def run_program():
    global cap, out, thread_pool

    # Load the YOLO model
    model = YOLO("yolov8s.pt")

    # Open the webcam feed
    source = "rtsp://admin:sec0mmth@192.168.254.3:554/Streaming/Channels/101"  # Add your stream sources here
    #source = "test_1.mp4"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Define the Modbus server address and port
    modbus_server_address = '192.168.254.66'
    modbus_server_port = 502  # Default Modbus TCP port

    # Define the server endpoint for HTTP frame upload
    server_url = "http://127.0.0.1:4321/upload_frame"

    def send_modbus_data(num_entered, num_left):
        with ModbusTcpClient(modbus_server_address, port=modbus_server_port) as client:
            if client.connect():
                print("Connected to Modbus server")

                address = 2000  # Start address for holding register
                values = [num_entered, num_left]
                registers = [int(val) for val in values]

                response = client.write_registers(address, registers)
                if response.isError():
                    print("Error writing to Modbus server")
                else:
                    print(f"Data sent to Modbus: Entered={num_entered}, Left={num_left}")

                # Delay for 3 seconds before resetting the counts
                threading.Timer(3, lambda: reset_modbus_counts(client, address)).start()

            else:
                print("Unable to connect to Modbus server")

    def reset_modbus_counts(client, address):
        values = [0, 0]
        registers = [int(val) for val in values]
        response = client.write_registers(address, registers)
        if response.isError():
            print("Error resetting Modbus counts")
        else:
            print("Modbus counts reset to 0, 0")

    def upload_frame(frame_bytes):
        headers = {'Content-Type': 'application/octet-stream'}
        try:
            response = requests.post(server_url, data=frame_bytes, headers=headers)
            if response.status_code != 200:
                print(f"Failed to send frame to server, status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending frame to server: {e}")

    # Get video properties
    original_w, original_h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Reduce frame size for processing
    processing_scale = 0.2
    w = int(original_w * processing_scale)
    h = int(original_h * processing_scale)
    w1 = int(w * 0.15)
    w2 = int(w * 0.85)
    h_top = int(h / 1.85)
    h_bot = int(h / 1.55)
    h_half = int(h / 1.77)
    region_points = [(w1, h_top), (w1, h_bot), (w2, h_bot), (w2, h_top)]
    region_points = [(w1, h_half), (w2, h_half)]

    # Initialize Object Counter
    counter = solutions.ObjectCounter(
        classes_names=model.names,
        reg_pts=region_points,
        view_img=True,
        draw_tracks=False,
    )

    # Process the video frames
    frame_skip = 15  # Adjust frame skip for real-time processing
    frame_count = 0
    class_want = 0

    # Define the directory to save recordings
    recordings_dir = 'recordings'
    os.makedirs(recordings_dir, exist_ok=True)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    now = datetime.now()
    video_filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
    video_path = os.path.join(recordings_dir, video_filename)
    out = cv2.VideoWriter(video_path, fourcc, fps // frame_skip, (w, h))

    # ThreadPool for HTTP requests
    thread_pool = ThreadPoolExecutor(max_workers=4)

    # Track the last update time and counters
    last_update_time = datetime.now()
    last_num_entered = 0
    last_num_left = 0

    try:
        while cap.isOpened():
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
                raise KeyboardInterrupt  # Raise a KeyboardInterrupt to handle clean exit
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            im0_resized = cv2.resize(im0, (w, h))
            tracks = model.track(im0_resized, persist=True, show=False, classes=class_want)

            im0_resized = counter.start_counting(im0_resized, tracks)

            num_entered = counter.in_counts
            num_left = counter.out_counts

            out.write(im0_resized)

            _, buffer = cv2.imencode('.jpg', im0_resized)
            frame_bytes = buffer.tobytes()
            thread_pool.submit(upload_frame, frame_bytes)

            # Check if the counters have changed
            if num_entered != last_num_entered or num_left != last_num_left:
                last_update_time = datetime.now()
                last_num_entered = num_entered
                last_num_left = num_left

            # Check if more than 3 seconds have passed without a counter change
            if datetime.now() - last_update_time > timedelta(seconds=2):
                if last_num_entered != 0 or last_num_left != 0:
                    thread_pool.submit(send_modbus_data, last_num_entered, last_num_left)
                    print(f"Data sent: Entered={last_num_entered}, Left={last_num_left}")
                    counter.reset_counts()
                    last_num_entered = 0
                    last_num_left = 0
                    last_update_time = datetime.now()

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
                raise KeyboardInterrupt  # Raise a KeyboardInterrupt to handle clean exit
    except KeyboardInterrupt:
        print("ESC key pressed. Exiting...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        thread_pool.shutdown(wait=True)

        print(f"Video saved: {video_path}")

# Function to check for exit command
def check_for_exit():
    while True:
        if input().strip().lower() == 'q':
            signal.raise_signal(signal.SIGINT)

# Start the exit-checking thread
exit_thread = threading.Thread(target=check_for_exit, daemon=True)
exit_thread.start()

# Run the main program loop
while True:
    run_program()
