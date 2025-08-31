import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # SORT tracker

# -------- Step 1: Load YOLOv8/YOLOv5 model --------
model = YOLO("yolov8n.pt")  # small & fast model (downloads automatically)

# -------- Step 2: Initialize SORT tracker --------
tracker = Sort()

# -------- Step 3: Open Webcam --------
cap = cv2.VideoCapture(0)  # use 0 for webcam, or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 4: Run YOLO detection
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.5:  # only keep confident detections
                detections.append([x1, y1, x2, y2, conf])

    # Convert to numpy array for SORT
    detections = np.array(detections)
    if len(detections) == 0:
        detections = np.empty((0, 5))

    # Step 5: Update tracker with detections
    tracked_objects = tracker.update(detections)

    # Step 6: Draw results
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO + SORT Object Tracking", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
