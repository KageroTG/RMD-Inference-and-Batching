import cv2
import json
from ultralytics import YOLO
from datetime import timedelta

# ==== CONFIGURATION ====
VIDEO_PATH = "Downloaded/10mins(part2).mp4"        # path to your recorded video
MODEL_PATH = "Epan_Dataset V6 YOLO (3 class) Yolov11n_Datasetv6_nofrm.pt"               # path to your YOLO model (.pt)
OUTPUT_METADATA = "metadata.json"     # output file for detections
CONF_THRESHOLD = 0.5                  # confidence threshold
# ========================

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
detections = []

# ROI variables
roi_start = None
roi_end = None
roi_selected = False
temp_frame = None

# --- Mouse callback to select ROI ---
def draw_roi(event, x, y, flags, param):
    global roi_start, roi_end, roi_selected, temp_frame
    frame_copy = temp_frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = None

    elif event == cv2.EVENT_MOUSEMOVE and roi_start:
        cv2.rectangle(frame_copy, roi_start, (x, y), (255, 0, 0), 2)
        cv2.imshow("Select ROI", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        roi_selected = True
        cv2.rectangle(frame_copy, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)
        print(f"ROI selected: {roi_start} to {roi_end}")

# --- STEP 1: Select ROI before starting inference ---
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

temp_frame = frame.copy()
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", draw_roi)

print("Drag your mouse to select ROI region, then press 's' to start inference.")

while True:
    cv2.imshow("Select ROI", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and roi_selected:
        cv2.destroyWindow("Select ROI")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

x1_roi, y1_roi = roi_start
x2_roi, y2_roi = roi_end
if x1_roi > x2_roi:
    x1_roi, x2_roi = x2_roi, x1_roi
if y1_roi > y2_roi:
    y1_roi, y2_roi = y2_roi, y1_roi

print(f"Inference will only detect objects within: ({x1_roi}, {y1_roi}) - ({x2_roi}, {y2_roi})")

# --- STEP 2: Run Inference ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video_time = str(timedelta(seconds=int(current_frame / fps)))

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # Draw ROI boundary
    cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check if detection inside ROI
            if x1_roi <= x1 <= x2_roi and y1_roi <= y1 <= y2_roi:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                detections.append({
                    "timestamp": video_time,
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

    cv2.imshow("Inference with ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- STEP 3: Save metadata ---
with open(OUTPUT_METADATA, "w") as f:
    json.dump(detections, f, indent=4)

print(f"Inference complete. {len(detections)} detections saved to {OUTPUT_METADATA}")
