import cv2
import json
import time
from ultralytics import YOLO
from datetime import timedelta

# ==== CONFIGURATION ====
VIDEO_PATH = "Downloaded/10mins(part2).mp4"         # path to your recorded video (use forward slashes to avoid escape sequences)
MODEL_PATH = "Epan_Dataset V6 YOLO (3 class) Yolov11n_Datasetv6_nofrm.pt"                # path to your YOLO model (.pt)
OUTPUT_METADATA = "metadata.json"      # output file for detections
CONF_THRESHOLD = 0.5                   # confidence threshold (drag this up/down to reduce false positives)
SHOW_WINDOW = True                     # set False if you just want to process silently
# ========================

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time (in seconds) and convert to hh:mm:ss
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video_time = str(timedelta(seconds=int(current_frame / fps)))

    # Run inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # Draw detections on the frame and collect data
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save detection info
            detections.append({
                "timestamp": video_time,
                "class": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    # Show the frame with detections
    if SHOW_WINDOW:
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save detections to JSON
with open(OUTPUT_METADATA, "w") as f:
    json.dump(detections, f, indent=4)

print(f"Inference completed. {len(detections)} detections saved to {OUTPUT_METADATA}")
