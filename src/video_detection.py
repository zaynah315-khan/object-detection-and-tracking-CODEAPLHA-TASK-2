import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize Deep SORT
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.3
)

video_path = "input_videos/test.mp4"
output_path = "output/tracked_video.mp4"

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit("Error: Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 25,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise SystemExit("Error: Could not create output video.")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Detect objects using YOLO
        results = model.predict(
            frame,
            conf=0.4,
            imgsz=416,
            verbose=False
        )

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                confidence,
                class_name
            ))

        # Track detected objects
        tracks = tracker.update_tracks(
            detections,
            frame=frame
        )

        # Draw boxes and tracking IDs
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            class_name = track.get_det_class() or "Object"
            label = f"{class_name} ID: {track_id}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        writer.write(frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

finally:
    cap.release()
    writer.release()

print("\nObject tracking completed!")
print(f"Total frames processed: {frame_count}")
print(f"Saved: {output_path}")
