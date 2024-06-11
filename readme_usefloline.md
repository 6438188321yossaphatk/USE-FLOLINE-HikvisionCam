requirements_usefloline.txt = pip install แต่ละอย่างที่ต้องใช้

usefloline_cam.py = file detection + modbus + http
    video -> saved in recordings folder
    parameter ปรับกล้องให้ delay น้อยลง -> processing_scale(ลด resolution) and frame_skip(frame process detection) 
    model = YOLO("yolov8n.pt")
        ใช้ cpu น่อยสุดไปมากสุด (yolov8n , yolov8s, yolov8m , yolov8l)


start_useflolinevision.bat = file autostart program 
    ต้องเอาไปใส่ใน WINDOW STARTUP FOLDER --> (window + r) --> shell:startup -->enter --> ใส่ start_useflolinevision.bat -shortuct ไว้ในนั้นละก้ได้ละครับ