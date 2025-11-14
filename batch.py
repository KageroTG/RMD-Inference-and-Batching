import json
from datetime import timedelta

# Configurable parameters
METADATA_PATH = "metadata.json"
TIME_WINDOW = 20  # seconds
MIN_DETECTIONS = 7

def parse_time(t):
    """Convert '0:01:26' into seconds (int)."""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def load_detections(path):
    """Load detection list from JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def make_batches(detections):
    """Create 20-second batches, keep only those with >= 7 detections."""
    # Sort by timestamp
    detections.sort(key=lambda x: parse_time(x["timestamp"]))

    batches = []
    current_batch = []
    start_time = None

    for det in detections:
        ts = parse_time(det["timestamp"])
        if start_time is None:
            start_time = ts
            current_batch = [det]
            continue

        # Check time span
        if ts - start_time <= TIME_WINDOW:
            current_batch.append(det)
        else:
            # Evaluate current batch
            if len(current_batch) >= MIN_DETECTIONS:
                batches.append(current_batch)
            # Start a new batch
            current_batch = [det]
            start_time = ts

    # Final check
    if len(current_batch) >= MIN_DETECTIONS:
        batches.append(current_batch)

    return batches

def publish_batches(batches):
    """Print and optionally save valid batches."""
    if not batches:
        print(" No batch met the threshold (>= 7 detections in 20 seconds).")
        return

    for i, batch in enumerate(batches, 1):
        print(f"\n Batch {i} — {len(batch)} detections")
        print("-" * 60)
        for det in batch:
            print(f"[{det['timestamp']}] {det['class']} | conf={det['confidence']:.3f}")
        print("-" * 60)

    # Optionally, save results
    with open("batched_output.json", "w") as f:
        json.dump(batches, f, indent=4)
    print("\n Saved valid batches to 'batched_output.json'")

def main():
    print(" Reading metadata.json ...")
    detections = load_detections(METADATA_PATH)
    print(f"Loaded {len(detections)} detections.")

    print(f"Batching detections (window={TIME_WINDOW}s, min={MIN_DETECTIONS})...")
    batches = make_batches(detections)
    publish_batches(batches)

if __name__ == "__main__":
    main()
