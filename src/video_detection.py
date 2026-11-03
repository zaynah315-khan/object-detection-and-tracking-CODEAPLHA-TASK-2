 
import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
video_path = "input_videos/test.mp4"

# Create output folder
os.makedirs("output", exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    raise SystemExit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
writer = cv2.VideoWriter(
    "output/detected_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 25,
    (width, height)
)

frame_count = 0
last_detected_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Detect objects every second frame
    if frame_count % 2 == 0 or last_detected_frame is None:
        results = model.predict(
            frame,
            conf=0.4,
            imgsz=416,
            verbose=False
        )

        last_detected_frame = results[0].plot()

    # Keep the latest detection boxes visible
    writer.write(last_detected_frame)

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
writer.release()

print("\nVideo detection completed!")
print("Saved: output/detected_video.mp4")
