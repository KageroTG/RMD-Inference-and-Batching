import json
from datetime import timedelta

# Config
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

        if ts - start_time <= TIME_WINDOW:
            current_batch.append(det)
        else:
            if len(current_batch) >= MIN_DETECTIONS:
                batches.append((start_time, current_batch))
            current_batch = [det]
            start_time = ts

    if len(current_batch) >= MIN_DETECTIONS:
        batches.append((start_time, current_batch))

    return batches

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
<h2>📦 Batched Detection Results</h2>
<p>Each collapsible section shows a batch that met the threshold (≥7 detections in a 20-second window).</p>
"""

    if not batches:
        html += "<p><b>No valid batches found.</b></p>"
    else:
        for i, (start_time, batch) in enumerate(batches, 1):
            start_sec = start_time
            end_sec = start_time + TIME_WINDOW
            html += f"""
<details>
  <summary>Batch {i} (from {start_sec}s to {end_sec}s)
  <span class='caption'>— {len(batch)} detections</span></summary>
  <table>
    <tr><th>#</th><th>Timestamp</th><th>Class</th><th>Confidence</th><th>Bounding Box</th></tr>
"""
            for idx, det in enumerate(batch, 1):
                html += f"<tr><td>{idx}</td><td>{det['timestamp']}</td><td>{det['class']}</td><td>{det['confidence']:.3f}</td><td>{det['bbox']}</td></tr>"
            html += "</table></details>\n"

    html += """
</body>
</html>
"""
    return html

def save_html(content):
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"💾 HTML report saved as '{OUTPUT_HTML}'")

def main():
    print("🔍 Reading metadata.json ...")
    detections = load_detections(METADATA_PATH)
    print(f"Loaded {len(detections)} detections.")

    print(f"⏱️ Grouping detections (window={TIME_WINDOW}s, min={MIN_DETECTIONS})...")
    batches = make_batches(detections)
    print(f"Generated {len(batches)} valid batches.")

    html_content = generate_html(batches)
    save_html(html_content)

if __name__ == "__main__":
    main()
