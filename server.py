from flask import Flask, request, Response, jsonify
import os
import datetime
import cv2
import threading
import numpy as np
app = Flask(__name__)

# Directory to save the frames
SAVE_DIR = 'received_frames'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Initialize a variable to hold the latest frame
latest_frame = None
lock = threading.Lock()

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame
    try:
        frame_data = request.data  # Get the raw bytes from the request

        # Decode the frame
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Save the frame to the specified directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # Update the latest frame
        with lock:
            latest_frame = frame

        return jsonify({"status": "success", "filename": filename}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def generate_video_feed():
    global latest_frame
    while True:
        if latest_frame is not None:
            with lock:
                frame = latest_frame.copy()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Server is running. Send frames to /upload_frame and view the feed at /video_feed."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
