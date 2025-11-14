import json
from datetime import timedelta

# Configuration
METADATA_PATH = "metadata.json"
TIME_WINDOW = 20  # seconds
MIN_DETECTIONS = 7
OUTPUT_HTML = "batched_output.html"

def parse_time(t):
    """Convert '0:01:26' -> seconds"""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def load_detections(path):
    with open(path, "r") as f:
        return json.load(f)

def make_batches(detections):
    """Main 20s window batching, with per-class grouping."""
    detections.sort(key=lambda x: parse_time(x["timestamp"]))
    batches = []
    start_time = None
    current_window = []

    for det in detections:
        ts = parse_time(det["timestamp"])
        if start_time is None:
            start_time = ts
            current_window = [det]
            continue

        if ts - start_time <= TIME_WINDOW:
            current_window.append(det)
        else:
            # When window ends, process and reset
            batch_data = classify_and_filter(current_window, start_time)
            if batch_data:
                batches.append((start_time, batch_data))
            start_time = ts
            current_window = [det]

    # Final batch
    batch_data = classify_and_filter(current_window, start_time)
    if batch_data:
        batches.append((start_time, batch_data))

    return batches

def classify_and_filter(window_detections, start_time):
    """Group detections by class and filter only those with enough counts."""
    class_groups = {}
    for det in window_detections:
        cls = det["class"]
        class_groups.setdefault(cls, []).append(det)

    # Filter classes below threshold
    valid_classes = {cls: dets for cls, dets in class_groups.items() if len(dets) >= MIN_DETECTIONS}
    return valid_classes if valid_classes else None

def generate_html(batches):
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Batched Detection Results</title>
<style>
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #f7f8fa;
  margin: 20px;
  color: #333;
}
summary {
  font-size: 1.1em;
  font-weight: 600;
  cursor: pointer;
  background-color: #e3e7ed;
  padding: 8px;
  border-radius: 6px;
  margin-bottom: 5px;
}
details {
  margin-bottom: 10px;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 6px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
}
th, td {
  border: 1px solid #ddd;
  padding: 6px 10px;
  text-align: left;
}
th {
  background-color: #0078d4;
  color: white;
}
tr:nth-child(even) {
  background-color: #f2f2f2;
}
.caption {
  font-style: italic;
  color: #555;
  margin-left: 5px;
}
</style>
</head>
<body>
<h2> Batched Detection Results by Class</h2>
<p>Grouped into 20-second windows; only classes with ≥{MIN_DETECTIONS} detections per window are included.</p>
"""

    if not batches:
        html += "<p><b>No valid class batches found.</b></p>"
    else:
        for i, (start_time, class_batches) in enumerate(batches, 1):
            end_time = start_time + TIME_WINDOW
            html += f"""
<details>
  <summary> Batch {i} (from {start_time}s to {end_time}s)</summary>
"""
            for cls_name, dets in class_batches.items():
                html += f"""
  <details style="margin-left: 20px;">
    <summary>Class: <b>{cls_name}</b> — {len(dets)} detections</summary>
    <table>
      <tr><th>#</th><th>Timestamp</th><th>Confidence</th><th>Bounding Box</th></tr>
"""
                for idx, det in enumerate(dets, 1):
                    html += f"<tr><td>{idx}</td><td>{det['timestamp']}</td><td>{det['confidence']:.3f}</td><td>{det['bbox']}</td></tr>"
                html += "    </table>\n  </details>\n"
            html += "</details>\n"

    html += "</body></html>"
    return html

def save_html(content):
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(content)
    print(f" HTML report saved as '{OUTPUT_HTML}'")

def main():
    print(" Reading metadata.json ...")
    detections = load_detections(METADATA_PATH)
    print(f"Loaded {len(detections)} detections.")

    print(f" Grouping detections (window={TIME_WINDOW}s, min={MIN_DETECTIONS})...")
    batches = make_batches(detections)
    print(f"Generated {len(batches)} valid time-window batches.")

    html_content = generate_html(batches)
    save_html(html_content)

if __name__ == "__main__":
    main()
