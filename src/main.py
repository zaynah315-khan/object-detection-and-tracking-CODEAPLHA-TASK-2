
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in the sample image
results = model.predict(
    source="bus.jpg",
    conf=0.4,
    save=True,
    project="output",
    name="image_detection",
    exist_ok=True
)

print("Object detection completed successfully!")
print("Check the output folder for the result.")

