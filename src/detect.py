import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# Project paths

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "yolov8n.pt"
VIDEO_PATH = ROOT_DIR / "videos" / "sample1.mp4"
OUTPUT_PATH = ROOT_DIR / "output" / "annotated" / "phase2_result.mp4"

# Detection settings

CONFIDENCE_THRESHOLD = 0.40
# CONFIDENCE_THRESHOLD = 0.25

# COCO class IDs
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Load model

model = YOLO(str(MODEL_PATH))

# Open video

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# Video properties

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)

if input_fps <= 0:
    input_fps = 30

# Video writer

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    input_fps,
    (width, height)
)

# FPS calculation

previous_time = time.time()

# Main detection loop

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO inference
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # Draw detections
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Ignore irrelevant classes
            if class_id not in TARGET_CLASSES:
                continue

            class_name = TARGET_CLASSES[class_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Label
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # Calculate FPS

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time


    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display frame

    cv2.imshow("EdgeAI - Object Detection", frame)

    # Save frame
    writer.write(frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to: {OUTPUT_PATH}")