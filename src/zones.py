"""
Edge AI for Smart City Surveillance - Phase 4: Counting & Zone Logic
Monitors defined crowd areas, evaluates overcrowding thresholds,
detects restricted-zone intrusions, and counts virtual line crossings.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.logger import SurveillanceLogger
from src.utils import (
    LineCrossingTracker,
    draw_hud,
    draw_text_box,
    draw_transparent_polygon,
    load_config,
    point_in_polygon,
)


def run_surveillance():
    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------
    cfg = load_config()

    model_path = Path(cfg["model"]["path"])
    conf_thresh = cfg["model"].get("confidence", 0.25)
    tracker_type = cfg["model"].get("tracker", "bytetrack.yaml")
    device = cfg["model"].get("device", "cpu")

    target_classes = {int(k): v for k, v in cfg["classes"].items()}

    video_input = Path(cfg["video"]["input_path"])
    output_dir = Path(cfg["video"]["output_dir"])
    output_path = output_dir / "phase4_result.mp4"
    show_display = cfg["video"].get("show_display", False)
    max_frames = cfg["video"].get("max_frames", None)

    # Crowd Zone configuration
    crowd_cfg = cfg["crowd_zone"]
    crowd_poly = np.array(crowd_cfg["polygon"], dtype=np.int32)
    crowd_threshold = crowd_cfg.get("threshold", 6)
    crowd_target_classes = set(crowd_cfg.get("target_classes", ["person"]))
    crowd_color = tuple(crowd_cfg.get("color", [0, 220, 0]))
    crowd_alpha = crowd_cfg.get("alpha", 0.22)

    # Restricted Zone configuration
    restr_cfg = cfg["restricted_zone"]
    restr_poly = np.array(restr_cfg["polygon"], dtype=np.int32)
    restr_alert_classes = set(restr_cfg.get("alert_classes", ["person", "bicycle"]))
    restr_color = tuple(restr_cfg.get("color", [0, 0, 220]))
    restr_alpha = restr_cfg.get("alpha", 0.20)

    # Counting Line configuration
    line_cfg = cfg["counting_line"]
    line_start = tuple(line_cfg["start"])
    line_end = tuple(line_cfg["end"])
    line_target_classes = line_cfg.get("target_classes", ["person"])
    line_color = tuple(line_cfg.get("color", [0, 255, 255]))
    line_thickness = line_cfg.get("thickness", 4)
    label_pos = line_cfg.get("label_positive", "In")
    label_neg = line_cfg.get("label_negative", "Out")

    line_tracker = LineCrossingTracker(
        line_start=line_start,
        line_end=line_end,
        target_classes=line_target_classes,
        padding=100,
    )

    # Initialize Phase 5 Surveillance Logger (SQLite & CSV)
    logger = SurveillanceLogger()

    print("=" * 65)
    print("PHASE 4: SURVEILLANCE INTELLIGENCE & ZONE ANALYTICS")
    print("=" * 65)
    print(f"Model              : {model_path}")
    print(f"Input Video        : {video_input}")
    print(f"Output Video       : {output_path}")
    print(f"Crowd Zone         : {crowd_cfg['name']} (Threshold: {crowd_threshold})")
    print(f"Restricted Zone    : {restr_cfg['name']} (Alert: {restr_alert_classes})")
    print(f"Counting Line      : {line_start} -> {line_end}")
    print("=" * 65)

    # ------------------------------------------------------------
    # Initialize YOLO Model
    # ------------------------------------------------------------
    model = YOLO(str(model_path))

    # ------------------------------------------------------------
    # Open Video Capture
    # ------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer: {output_path}")

    # Scale multiplier for annotations based on frame height
    scale = width / 1920.0

    frame_idx = 0
    prev_time = time.time()
    overcrowding_event_count = 0
    restricted_intrusion_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # --------------------------------------------------------
        # Multi-Object Tracking with ByteTrack
        # --------------------------------------------------------
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_type,
            conf=conf_thresh,
            device=device,
            verbose=False,
        )

        result = results[0]

        total_people = 0
        crowd_zone_people = 0
        restricted_violations = 0
        active_intrusions = []
        current_restricted_tids = set()

        # --------------------------------------------------------
        # Draw Zones and Line on Frame
        # --------------------------------------------------------
        # 1. Crowd Zone
        draw_transparent_polygon(
            frame,
            crowd_poly,
            color=crowd_color,
            alpha=crowd_alpha,
            border_thickness=int(3 * scale),
        )

        # Crowd Zone Header Label
        cw_label_pos = (crowd_poly[0][0] + 15, crowd_poly[0][1] - int(15 * scale))
        draw_text_box(
            frame,
            f"CROWD ZONE: {crowd_cfg['name']}",
            cw_label_pos,
            font_scale=0.7 * scale,
            text_color=(0, 255, 0),
            bg_color=(20, 20, 20),
            thickness=int(2 * scale),
        )

        # 2. Restricted Zone
        draw_transparent_polygon(
            frame,
            restr_poly,
            color=restr_color,
            alpha=restr_alpha,
            border_thickness=int(3 * scale),
        )

        # Restricted Zone Header Label
        restr_label_pos = (restr_poly[0][0] + 15, restr_poly[0][1] - int(15 * scale))
        draw_text_box(
            frame,
            f"RESTRICTED ZONE: {restr_cfg['name']}",
            restr_label_pos,
            font_scale=0.7 * scale,
            text_color=(0, 0, 255),
            bg_color=(20, 20, 20),
            thickness=int(2 * scale),
        )

        # 3. Virtual Counting Line
        cv2.line(
            frame,
            line_start,
            line_end,
            line_color,
            int(line_thickness * scale),
            cv2.LINE_AA,
        )
        line_mid_y = (line_start[1] + line_end[1]) // 2
        line_mid_x = (line_start[0] + line_end[0]) // 2
        draw_text_box(
            frame,
            f"COUNTING LINE [{label_pos} / {label_neg}]",
            (line_mid_x + 15, line_mid_y),
            font_scale=0.65 * scale,
            text_color=line_color,
            bg_color=(30, 30, 30),
            thickness=int(2 * scale),
        )

        # --------------------------------------------------------
        # Process Tracked Detections
        # --------------------------------------------------------
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()

            for track_id, class_id, conf, box in zip(
                track_ids, class_ids, confidences, coordinates
            ):
                if class_id not in target_classes:
                    continue

                class_name = target_classes[class_id]
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                center = (cx, cy)

                # Total People counter
                if class_name == "person":
                    total_people += 1

                # ------------------------------------------------
                # A. Line Crossing Tracker
                # ------------------------------------------------
                event = line_tracker.update(track_id, center, class_name)
                if event:
                    direction_label = label_pos if event == "in" else label_neg
                    print(
                        f"[Frame {frame_idx:03d}] Line Crossing: {class_name} "
                        f"ID:{track_id} -> {direction_label}"
                    )
                    logger.log_line_crossing(
                        frame_idx=frame_idx,
                        object_class=class_name,
                        track_id=track_id,
                        direction=direction_label,
                        x=cx,
                        y=cy,
                        confidence=conf,
                    )

                # ------------------------------------------------
                # B. Crowd Zone Evaluation
                # ------------------------------------------------
                in_crowd = False
                if class_name in crowd_target_classes:
                    if point_in_polygon(center, crowd_poly):
                        crowd_zone_people += 1
                        in_crowd = True

                # ------------------------------------------------
                # C. Restricted Zone Evaluation
                # ------------------------------------------------
                in_restricted = False
                if class_name in restr_alert_classes:
                    if point_in_polygon(center, restr_poly):
                        restricted_violations += 1
                        in_restricted = True
                        current_restricted_tids.add(track_id)
                        active_intrusions.append((class_name, track_id, box))
                        logger.log_restricted_intrusion(
                            frame_idx=frame_idx,
                            object_class=class_name,
                            track_id=track_id,
                            zone_name=restr_cfg["name"],
                            x=cx,
                            y=cy,
                            confidence=conf,
                        )

                # ------------------------------------------------
                # Visual Rendering for Tracked Object
                # ------------------------------------------------
                # Choose color based on state
                if in_restricted:
                    box_color = (0, 0, 255)  # Red for restricted intrusion
                elif in_crowd:
                    box_color = (0, 255, 0)  # Green for crowd zone
                else:
                    box_color = (255, 180, 50)  # Blue-orange for standard

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    int(2 * scale),
                )

                # Draw center point
                cv2.circle(
                    frame,
                    center,
                    int(5 * scale),
                    (0, 255, 255),
                    -1,
                )

                # Draw label
                obj_label = f"{class_name} ID:{track_id} {conf:.2f}"
                draw_text_box(
                    frame,
                    obj_label,
                    (x1, max(y1 - 10, 20)),
                    font_scale=0.55 * scale,
                    text_color=(255, 255, 255),
                    bg_color=box_color,
                    thickness=int(2 * scale),
                    padding=int(4 * scale),
                )

                # If in restricted zone, draw prominent intrusion tag
                if in_restricted:
                    draw_text_box(
                        frame,
                        f"RESTRICTED INTRUSION [ID:{track_id}]",
                        (x1, min(y2 + int(25 * scale), height - 10)),
                        font_scale=0.65 * scale,
                        text_color=(255, 255, 255),
                        bg_color=(0, 0, 255),
                        thickness=int(2 * scale),
                        padding=int(4 * scale),
                    )

        # Update restricted zone tracked objects for debouncing
        logger.update_restricted_tracks(current_restricted_tids)

        # --------------------------------------------------------
        # Overcrowding Logic
        # --------------------------------------------------------
        # Crucial: Evaluated ONLY on people inside the crowd zone!
        overcrowding = crowd_zone_people >= crowd_threshold
        logger.log_crowd_status(
            frame_idx=frame_idx,
            count=crowd_zone_people,
            threshold=crowd_threshold,
            zone_name=crowd_cfg["name"],
        )

        if overcrowding:
            overcrowding_event_count += 1

            # Render prominent Overcrowding Alert directly over the crowd zone
            alert_x = crowd_poly[0][0] + int(30 * scale)
            alert_y = crowd_poly[0][1] + int(60 * scale)
            draw_text_box(
                frame,
                f"!! OVERCROWDING ALERT !! ({crowd_zone_people} People)",
                (alert_x, alert_y),
                font_scale=1.1 * scale,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 255),
                thickness=int(3 * scale),
                padding=int(10 * scale),
            )

        if restricted_violations > 0:
            restricted_intrusion_count += restricted_violations

        # --------------------------------------------------------
        # Surveillance HUD Overlay
        # --------------------------------------------------------
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        draw_hud(
            frame=frame,
            fps=fps,
            total_people=total_people,
            crowd_zone_people=crowd_zone_people,
            crowd_threshold=crowd_threshold,
            overcrowding=overcrowding,
            restricted_violations=restricted_violations,
            entry_count=line_tracker.in_count,
            exit_count=line_tracker.out_count,
            scale_factor=1.0,
        )

        # Write to video
        writer.write(frame)

        if show_display:
            # Downscale for display window if frame is 4K
            display_frame = cv2.resize(frame, (int(width * 0.35), int(height * 0.35)))
            cv2.imshow("Edge AI Smart City Surveillance", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 40 == 0:
            print(
                f"[Frame {frame_idx:03d}/{total_frames}] "
                f"FPS: {fps:4.1f} | "
                f"Total People: {total_people:2d} | "
                f"Crowd Zone: {crowd_zone_people:2d}/{crowd_threshold} "
                f"({'ALERT' if overcrowding else 'OK'}) | "
                f"Restricted: {restricted_violations:2d} | "
                f"Crossings [In:{line_tracker.in_count} Out:{line_tracker.out_count}]"
            )

    # ------------------------------------------------------------
    # Cleanup & Summary
    # ------------------------------------------------------------
    cap.release()
    writer.release()
    if show_display:
        cv2.destroyAllWindows()

    # Print database event log summary
    logger.print_recent_events(limit=15)
    db_summary = logger.get_summary()
    print("Database Event Counts:")
    for evt, count in db_summary.items():
        print(f"  - {evt:<18}: {count}")
    logger.close()

    print()
    print("=" * 65)
    print("PHASE 4 & 5 EXECUTION COMPLETE")
    print("=" * 65)
    print(f"Total Frames Processed     : {frame_idx}")
    print(f"Line Crossings [In / Out]  : {line_tracker.in_count} / {line_tracker.out_count}")
    print(f"Overcrowding Alert Frames  : {overcrowding_event_count}")
    print(f"Restricted Intrusion Frames: {restricted_intrusion_count}")
    print(f"Annotated Output Saved To  : {output_path}")
    print(f"Database Saved To          : {logger.db_path}")
    print(f"CSV Logs Saved To          : {logger.csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_surveillance()