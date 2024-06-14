
# USE-FLOLINE PeopleCouting  IP-cam

A python program to use with an ip camera to count numbers of people entering and leaving the building using opencv and yolov8 for object detection and tracking with a Modbus TCP communication.


## Screenshots
![Alt Text](https://i.ibb.co/Xk6jK5h/Image-14-6-2567-BE-at-09-53.jpg)



## Features

- Real-time object detection and tracking using YOLOv8.
- Counting objects that enter and leave a defined region.
- Modbus TCP integration to send the count of objects to a PLC.
- HTTP frame upload to a specified server endpoint.
- Video recording of the processed feed.
- Graceful shutdown handling to ensure resources are released properly.



## Requirements

- Python 3.8 or higher
- Required Python libraries:
    - opencv-python
    - ultralytics
    - requests
    - pymodbus



## Installation

1. Clone repository in github to VS Code

```bash
git clone https://github.com/6438188321yossaphatk/USE-FLOLINE.git
cd real-time-object-detection

```
    
2. Install required packages:

```bash
pip install -r requirements.txt
```
## Usage

1. Run the program:
```bash
#cd into the directory containing this file first
python usefloline_cam.py
```

2. Stream URL :
- The code in usefloline_cam.py currently uses a predefined RTSP stream URL. Replace the ip address with your own source URL.
```bash
source = "rtsp://admin:sec0mmth@192.168.254.3:554/Streaming/Channels/101" 
```

3. Interact with program:
- The program will start processing frames from the video source.
- It counts objects crossing defined regions and updates these counts to a Modbus server.
- To exit the program, Press Ctrl+C  in the terminal until it stop and exit the program. Otherwise, the programm will loop.
- The saved videos will be in the directory called "recordings" in which the video is ordered by the datetime it is saved.
