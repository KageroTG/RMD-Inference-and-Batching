import cv2
import torch
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

class RoadDefectDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the Road Defect Detector
        
        Args:
            model_path: Path to the .pt model file
            confidence_threshold: Minimum confidence score for detections
        """
        self.detections = []
        self.confidence_threshold = confidence_threshold
        
        # Try multiple methods to load the model
        self.model = self._load_model(model_path)
        self.class_names = self._get_class_names()
        
        print(f"Model loaded successfully!")
        print(f"Available classes: {self.class_names}")
    
    def _load_model(self, model_path):
        """Load model with multiple fallback methods"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Method 1: Try direct torch load first
        try:
            print("Attempting to load model with torch.load...")
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'eval'):
                model.eval()
            print("✓ Model loaded successfully using torch.load")
            return model
        except Exception as e:
            print(f"✗ torch.load failed: {e}")
        
        # Method 2: Try Ultralytics YOLO
        try:
            print("Attempting to load model with Ultralytics YOLO...")
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✓ Model loaded successfully using Ultralytics YOLO")
            return model
        except ImportError:
            print("✗ Ultralytics not available")
        except Exception as e:
            print(f"✗ Ultralytics YOLO failed: {e}")
        
        # Method 3: Try torch.hub (original method)
        try:
            print("Attempting to load model with torch.hub...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
            print("✓ Model loaded successfully using torch.hub")
            return model
        except Exception as e:
            print(f"✗ torch.hub failed: {e}")
        
        raise RuntimeError("All model loading methods failed. Please check your model file and dependencies.")
    
    def _get_class_names(self):
        """Extract class names from the model"""
        if hasattr(self.model, 'names'):
            return self.model.names
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
            return self.model.module.names
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
            return self.model.model.names
        else:
            # Default class names based on your requirements
            print("⚠ Warning: Using default class names")
            return {0: 'frm', 1: 'potholes', 2: 'raveling', 3: 'cracks'}
    
    def _get_color(self, class_name):
        """Get consistent color for each class"""
        color_map = {
            'frm': (0, 255, 255),      # Yellow
            'potholes': (0, 0, 255),   # Red
            'raveling': (255, 0, 0),   # Blue
            'cracks': (0, 255, 0),     # Green
        }
        return color_map.get(class_name.lower(), (255, 255, 255))  # Default white
    
    def process_video(self, video_path, output_path=None):
        """
        Process video and detect road defects
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
        """
        # Check if video file exists
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time_seconds = frame_count / fps
            
            # Convert seconds to hours, minutes, seconds
            hours = int(current_time_seconds // 3600)
            minutes = int((current_time_seconds % 3600) // 60)
            seconds = int(current_time_seconds % 60)
            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Perform inference based on model type
            if hasattr(self.model, 'predict'):  # Ultralytics YOLO
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                detections_frame = self._process_ultralytics_results(results, frame_count, timestamp, frame)
            else:  # Standard YOLOv5
                # Set confidence threshold for YOLOv5
                if hasattr(self.model, 'conf'):
                    self.model.conf = self.confidence_threshold
                results = self.model(frame)
                detections_frame = self._process_yolov5_results(results, frame_count, timestamp, frame)
            
            # Add timestamp and info to frame
            cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detections_frame)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            # Display progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        processing_time = time.time() - start_time
        print(f"Video processing completed in {processing_time:.2f} seconds")
        print(f"Total detections: {len(self.detections)}")
    
    def _process_yolov5_results(self, results, frame_count, timestamp, frame):
        """Process results from YOLOv5 model"""
        detections_frame = []
        
        # Handle different YOLOv5 result formats
        if hasattr(results, 'xyxy'):
            boxes = results.xyxy[0]
        elif hasattr(results, 'pred'):
            boxes = results.pred[0]
        else:
            boxes = results[0] if isinstance(results, (list, tuple)) else results
        
        for detection in boxes:
            if len(detection) >= 6:  # x1, y1, x2, y2, conf, class
                x1, y1, x2, y2, conf, cls = detection[:6]
            else:
                continue
                
            confidence = float(conf)
            if confidence < self.confidence_threshold:
                continue
                
            class_id = int(cls)
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Store detection
            detection_data = {
                "timestamp": timestamp,
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                "frame": frame_count
            }
            
            self.detections.append(detection_data)
            detections_frame.append(detection_data)
            
            # Draw bounding box and label with class-specific color
            label = f"{class_name} {confidence:.2f}"
            color = self._get_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections_frame
    
    def _process_ultralytics_results(self, results, frame_count, timestamp, frame):
        """Process results from Ultralytics YOLO model"""
        detections_frame = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                confidence = float(box.conf)
                if confidence < self.confidence_threshold:
                    continue
                    
                class_id = int(box.cls)
                class_name = self.class_names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Store detection
                detection_data = {
                    "timestamp": timestamp,
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "frame": frame_count
                }
                
                self.detections.append(detection_data)
                detections_frame.append(detection_data)
                
                # Draw bounding box and label with class-specific color
                label = f"{class_name} {confidence:.2f}"
                color = self._get_color(class_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections_frame
    
    def save_metadata(self, output_path):
        """
        Save detection metadata to JSON file
        
        Args:
            output_path: Path to save metadata JSON file
        """
        metadata = {
            "processing_date": datetime.now().isoformat(),
            "total_detections": len(self.detections),
            "detections_by_class": self._count_detections_by_class(),
            "detections": self.detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {output_path}")
    
    def _count_detections_by_class(self):
        """Count detections by class"""
        class_counts = {}
        for detection in self.detections:
            class_name = detection["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

def main():
    # Configuration
    MODEL_PATH = "Epan_Dataset V6 YOLO (3 class) Yolov11n_Datasetv6_nofrm.pt"
    VIDEO_PATH = "Downloaded/720p KL Road Full.mp4"
    OUTPUT_VIDEO = "inferenceOutput_video.mp4"
    METADATA_JSON = "metadata.json"
    CONFIDENCE_THRESHOLD = 0.5
    
    # Check if files exist
    if not Path(MODEL_PATH).exists():
        print(f" Error: Model file '{MODEL_PATH}' not found!")
        print("Please check the path and try again.")
        return
    
    if not Path(VIDEO_PATH).exists():
        print(f" Error: Video file '{VIDEO_PATH}' not found!")
        print("Please check the path and try again.")
        return
    
    print(" Initializing Road Defect Detector...")
    try:
        detector = RoadDefectDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    except Exception as e:
        print(f" Failed to initialize detector: {e}")
        print("\n Installation tips:")
        print("1. Install missing dependencies: pip install ultralytics")
        print("2. Make sure your model file is compatible")
        return
    
    try:
        print(" Starting video processing...")
        detector.process_video(VIDEO_PATH, OUTPUT_VIDEO)
        
        print(" Saving metadata...")
        detector.save_metadata(METADATA_JSON)
        
        print("\n" + "="*60)
        print(" Processing completed successfully!")
        print(f" Output video: {OUTPUT_VIDEO}")
        print(f" Metadata: {METADATA_JSON}")
        print("="*60)
        
    except Exception as e:
        print(f" Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()