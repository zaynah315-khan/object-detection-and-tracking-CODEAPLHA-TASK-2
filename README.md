# Object Detection and Tracking – CodeAlpha Internship task 4

## Project Overview
This project uses YOLO11 and Deep SORT to detect and track objects in a video.

## Technologies Used
- Python
- YOLO11 (Ultralytics)
- Deep SORT
- OpenCV
- PyTorch

## Features
- Detects objects such as people and buses.
- Draws bounding boxes around detected objects.
- Assigns tracking IDs to detected objects.
- Processes video frame by frame.
- Saves the tracked video as an MP4 file.

## Project Structure
```text
object-detection-and-tracking/
├── input_videos/
│   └── test.mp4
├── output/
│   └── tracked_video.mp4
├── screenshots/
├── src/
│   ├── main.py
│   └── video_detection.py
├── yolo11n.pt
└── README.md
```

## Installation

Install the required libraries:

```bash
python -m pip install ultralytics opencv-python deep-sort-realtime "setuptools<81"
```

## How to Run

Run image detection:

```bash
python src/main.py
```

Run video detection and tracking:

```bash
python src/video_detection.py
```

## Output
The processed video is saved at:

`output/tracked_video.mp4`

The output displays detected objects with bounding boxes, class labels, and tracking IDs.

## Internship
CodeAlpha Artificial Intelligence Internship – Task 4: Object Detection and Tracking.
