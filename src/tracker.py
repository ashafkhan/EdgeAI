"""
Edge AI for Smart City Surveillance - Phase 3 Object Tracking
Performs persistent multi-object tracking with YOLOv8n and ByteTrack.
"""

from pathlib import Path
import time
import cv2
from ultralytics import YOLO

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "yolov8n.pt"
VIDEO_PATH = ROOT_DIR / "videos" / "sample1.mp4"
OUTPUT_PATH = ROOT_DIR / "output" / "annotated" / "phase3_result.mp4"

CONFIDENCE_THRESHOLD = 0.25

# Target smart-city COCO classes
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color palette for classes (BGR)
CLASS_COLORS = {
    "person": (255, 144, 30),      # Blue/Orange
    "bicycle": (0, 215, 255),      # Gold
    "car": (0, 255, 128),          # Spring Green
    "motorcycle": (255, 0, 255),   # Magenta
    "bus": (0, 165, 255),          # Orange
    "truck": (255, 255, 0),        # Cyan
}


def run_tracker():
    print("=" * 60)
    print("PHASE 3: MULTI-OBJECT TRACKING (YOLOv8n + ByteTrack)")
    print("=" * 60)
    print(f"Model  : {MODEL_PATH}")
    print(f"Video  : {VIDEO_PATH}")
    print(f"Output : {OUTPUT_PATH}")

    # Load YOLOv8n model
    model = YOLO(str(MODEL_PATH))

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

    # Maintain visual trail history for active tracks
    track_history = {}
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO with ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()

            for track_id, class_id, conf, box in zip(
                track_ids, class_ids, confidences, coordinates
            ):
                if class_id not in TARGET_CLASSES:
                    continue

                class_name = TARGET_CLASSES[class_id]
                color = CLASS_COLORS.get(class_name, (0, 255, 0))
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Store trajectory trail
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Draw track trail
                points = track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], color, 2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                # Draw label with track ID
                label = f"{class_name} ID:{track_id} ({conf:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, max(y1 - lh - 10, 0)), (x1 + lw + 10, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # FPS calculation
        curr_time = time.time()
        curr_fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Display tracking info overlay
        cv2.putText(
            frame,
            f"ByteTrack Active | FPS: {curr_fps:.1f} | Frame: {frame_count}/{total_frames}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({curr_fps:.1f} FPS)...")

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print("PHASE 3 COMPLETE")
    print(f"Tracked video saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run_tracker()
