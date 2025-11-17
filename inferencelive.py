import cv2
import json
import time
import os
from ultralytics import YOLO
from datetime import timedelta, datetime

# ==== CONFIGURATION ====
WEBCAM_INDEX = 0                       # Webcam index (0 for default, 1 for second camera, etc.)
MODEL_PATH = "yolo11nC4V6_openvino_model"  # path to your YOLO model (.pt) or OpenVINO model directory
# Alternative: MODEL_PATH = "Epan_Dataset V6 YOLO (3 class) Yolov11n_Datasetv6_nofrm.pt"  # Use this for .pt format
OUTPUT_METADATA = "metadataLive.json"      # output file for detections
CONF_THRESHOLD = 0.5                   # confidence threshold (drag this up/down to reduce false positives)
SHOW_WINDOW = True                     # set False if you just want to process silently
BATCH_SIZE = 4                         # Number of frames to process together (adjust based on GPU memory)
# ========================

# Load the YOLO model (supports both .pt and OpenVINO formats)
print(f"Loading model from: {MODEL_PATH}")
if os.path.isdir(MODEL_PATH) or MODEL_PATH.endswith('/'):
    # OpenVINO model format (directory)
    print("Detected OpenVINO model format")
    model = YOLO(MODEL_PATH)
else:
    # Standard .pt model format
    print("Detected standard PyTorch model format (.pt)")
    model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# Open the webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_INDEX}.")
    print("Trying webcam 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open any webcam.")
        exit()

# Set webcam properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Get actual webcam FPS (may vary)
webcam_fps = cap.get(cv2.CAP_PROP_FPS)
if webcam_fps <= 0:
    webcam_fps = 60  # Default fallback

detections = []
frame_batch = []  # Store frames for batching
frame_metadata = []  # Store frame numbers and timestamps

# Performance tracking for live view
fps_start_time = time.time()
session_start_time = time.time()  # Track session start for elapsed time
fps_frame_count = 0
current_fps = 0
frame_counter = 0  # Track total frames processed

def draw_text_with_background(img, text, position, font_scale=0.6, thickness=2, 
                               text_color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.7):
    """Draw text with semi-transparent background for better readability"""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                           font_scale, thickness)
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - text_height - 5), 
                  (x + text_width + 5, y + baseline + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness)
    return img

def process_batch(batch_frames, batch_metadata):
    """Process a batch of frames and return detections"""
    global fps_frame_count, current_fps, fps_start_time
    
    batch_detections = []
    
    # Run batched inference with stream=True
    results = model.track(batch_frames, conf=CONF_THRESHOLD, verbose=False, persist=True, stream=True)
    
    # Process results in order (stream=True yields results as they complete)
    for idx, result in enumerate(results):
        video_time = batch_metadata[idx]['timestamp']
        frame_num = batch_metadata[idx]['frame_num']
        original_frame = batch_frames[idx].copy()
        
        boxes = result.boxes
        detection_count = len(boxes)
        
        # Draw detections
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get track ID (if tracking is enabled)
            track_id = int(box.id[0]) if box.id is not None else None
            
            # Draw bounding box and label
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_text = f"{label} {conf:.2f}"
            if track_id is not None:
                display_text += f" ID:{track_id}"
            cv2.putText(original_frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save detection info
            detection_data = {
                "timestamp": video_time,
                "class": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            }
            if track_id is not None:
                detection_data["track_id"] = track_id
            batch_detections.append(detection_data)
        
        # Calculate FPS
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 0:
            current_fps = fps_frame_count / elapsed_time
        
        # Draw live view information overlay
        if SHOW_WINDOW:
            # Calculate elapsed time since session start
            elapsed_seconds = time.time() - session_start_time
            elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
            
            # Information text lines
            info_lines = [
                f"Webcam: {WEBCAM_INDEX} | Frame: {frame_num}",
                f"Elapsed Time: {elapsed_time}",
                f"Processing FPS: {current_fps:.1f}",
                f"Detections: {detection_count}",
                f"Total Detections: {len(detections) + len(batch_detections)}",
                f"Batch Size: {BATCH_SIZE}"
            ]
            
            # Draw info overlay at top-left
            y_offset = 30
            for i, line in enumerate(info_lines):
                original_frame = draw_text_with_background(
                    original_frame, line, (10, y_offset + i * 30),
                    font_scale=0.7, thickness=2,
                    text_color=(0, 255, 0), bg_color=(0, 0, 0)
                )
            
            # Show the frame with detections
            cv2.imshow("Live Inference - Press 'q' to quit", original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return batch_detections, True  # Return True to signal early exit
    
    return batch_detections, False

print(f"Starting live inference from webcam {WEBCAM_INDEX}...")
print("Press 'q' to quit and save detections.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame from webcam. Retrying...")
        time.sleep(0.1)
        continue
    
    # Increment frame counter and get real-time timestamp
    frame_counter += 1
    current_time = time.time()
    elapsed_seconds = current_time - session_start_time
    video_time = str(timedelta(seconds=int(elapsed_seconds)))
    
    # Add frame to batch
    frame_batch.append(frame)
    frame_metadata.append({
        'frame_num': frame_counter,
        'timestamp': video_time
    })
    
    # Process batch when it reaches BATCH_SIZE
    if len(frame_batch) >= BATCH_SIZE:
        batch_detections, should_break = process_batch(frame_batch, frame_metadata)
        detections.extend(batch_detections)
        
        if should_break:
            break
        
        # Clear batch for next iteration
        frame_batch = []
        frame_metadata = []

# Process any remaining frames in batch before exiting
if frame_batch:
    print("Processing remaining frames...")
    batch_detections, _ = process_batch(frame_batch, frame_metadata)
    detections.extend(batch_detections)

cap.release()
cv2.destroyAllWindows()

# Save detections to JSON
with open(OUTPUT_METADATA, "w") as f:
    json.dump(detections, f, indent=4)

print(f"\nLive inference completed. {len(detections)} detections saved to {OUTPUT_METADATA}")
print(f"Total frames processed: {frame_counter}")
