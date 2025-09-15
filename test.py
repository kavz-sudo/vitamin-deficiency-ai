from ultralytics import YOLO
import cv2
import time
import zipfile
import os



# 🔸 Load YOLOv8 model
# Use 'yolov8n.pt' or a custom model like 'best.pt' trained for ambulances
model = YOLO('yolov8n.pt')  # replace with 'best.pt' if custom-trained

# 🔸 Load traffic video
video_path = 'traffic_signal_video.mp4'  # path to your sample traffic video
cap = cv2.VideoCapture(video_path)
zip_path = "ambulance-dataset.zip"
extract_path = "/content/datasets"




# Check if video loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 🔸 Process video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # Resize frame for consistent detection
    resized_frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 detection
    results = model.predict(resized_frame, conf=0.5)  # confidence threshold

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Check for ambulance detection
    for result in results:
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            class_name = model.names[class_id]
            confidence = float(conf)

            # 🚨 If ambulance detected, trigger response
            if class_name.lower() == "ambulance":
                print(f"🚨 Ambulance detected with confidence {confidence:.2f}!")
                # 👉 TODO: Trigger GPIO or send signal to ESP32 if needed
                # e.g., serial.write(b'1\n') or send request to ESP32
                time.sleep(5)  # simulate action delay
    for detection in detections:
    if detection.class_name == 'ambulance':
        cx, cy = detection.center
        distance_to_center = euclidean_distance(cx, cy, img_width/2, img_height/2)
        # Keep the closest one

    # Show result
    cv2.imshow("Ambulance Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
# Create output dir and extract
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

    

