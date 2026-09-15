"""
Edge AI for Smart City Surveillance - Edge ONNX Inference Engine
Optimized for edge device deployment using ONNX Runtime (INT8 / FP32)
with live GUI display window, zone monitoring, line counting, and SQLite logging.
"""

import argparse
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


def run_edge_inference(
    model_path: str = None,
    source: str = None,
    display: bool = True,
    save_video: bool = True,
    conf_thresh: float = None,
    max_frames: int = None,
    loop: bool = True,
):
    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    cfg = load_config()

    if model_path is None:
        # Default to quantized INT8 ONNX model for edge deployment
        int8_path = ROOT_DIR / "models" / "yolov8n_int8.onnx"
        model_path = int8_path if int8_path.is_file() else ROOT_DIR / "models" / "yolov8n.onnx"
    else:
        model_path = Path(model_path)

    if source is None:
        source = cfg["video"]["input_path"]
    
    # Check if source is a camera index (e.g. "0")
    if str(source).isdigit():
        video_source = int(source)
        is_webcam = True
    else:
        video_source = str(source)
        is_webcam = False

    conf = conf_thresh or cfg["model"].get("confidence", 0.25)
    target_classes = {int(k): v for k, v in cfg["classes"].items()}

    # Output path
    output_dir = Path(cfg["video"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = "int8" if "int8" in str(model_path).lower() else "onnx"
    output_path = output_dir / f"edge_{model_tag}_result.mp4"

    # Zone definitions
    crowd_cfg = cfg["crowd_zone"]
    crowd_poly = np.array(crowd_cfg["polygon"], dtype=np.int32)
    crowd_threshold = crowd_cfg.get("threshold", 6)
    crowd_target_classes = set(crowd_cfg.get("target_classes", ["person"]))
    crowd_color = tuple(crowd_cfg.get("color", [0, 220, 0]))
    crowd_alpha = crowd_cfg.get("alpha", 0.22)

    restr_cfg = cfg["restricted_zone"]
    restr_poly = np.array(restr_cfg["polygon"], dtype=np.int32)
    restr_alert_classes = set(restr_cfg.get("alert_classes", ["person", "bicycle"]))
    restr_color = tuple(restr_cfg.get("color", [0, 0, 220]))
    restr_alpha = restr_cfg.get("alpha", 0.20)

    line_cfg = cfg["counting_line"]
    line_start = tuple(line_cfg["start"])
    line_end = tuple(line_cfg["end"])
    line_target_classes = line_cfg.get("target_classes", ["person"])
    line_color = tuple(line_cfg.get("color", [0, 255, 255]))
    label_pos = line_cfg.get("label_positive", "In")
    label_neg = line_cfg.get("label_negative", "Out")

    line_tracker = LineCrossingTracker(
        line_start=line_start,
        line_end=line_end,
        target_classes=line_target_classes,
        padding=100,
    )

    logger = SurveillanceLogger()

    print("\n" + "=" * 70)
    print("🚀 EDGE AI LIVE SURVEILLANCE ENGINE (ONNX RUNTIME)")
    print("=" * 70)
    print(f"Edge Model     : {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Video Source   : {video_source} {'(Live Webcam)' if is_webcam else '(Recorded CCTV)'}")
    print(f"Live GUI Window: {'ENABLED (Press Q to quit)' if display else 'DISABLED (Headless)'}")
    print(f"Save Annotated : {output_path if save_video else 'DISABLED'}")
    print(f"Crowd Limit    : {crowd_threshold} persons in {crowd_cfg['name']}")
    print(f"Restricted Area: {restr_cfg['name']} (Intrusion Alerts: {restr_alert_classes})")
    print(f"Playback Mode  : {'CONTINUOUS LOOP (Runs until Q or window closed)' if loop and not is_webcam else 'SINGLE PASS'}")
    print("=" * 70 + "\n")

    # Load ONNX model via Ultralytics ONNX Runtime backend
    model = YOLO(str(model_path), task="detect")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else "Live"

    writer = None
    if save_video and not is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    # Determine display window scale to fit typical laptop screens (target height ~850)
    disp_scale = min(1.0, 850.0 / float(height))
    disp_w = int(width * disp_scale)
    disp_h = int(height * disp_scale)

    if display:
        cv2.namedWindow("Edge AI - Live Smart City Surveillance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Edge AI - Live Smart City Surveillance", disp_w, disp_h)

    scale = width / 1920.0
    frame_idx = 0
    prev_time = time.time()
    overcrowd_frames = 0
    intrusion_frames = 0
    loop_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop and not is_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                loop_count += 1
                print(f"\n🔁 [Loop {loop_count}] Video completed. Seamlessly looping playback for continuous demonstration...\n")
                # Finalize video writer after first loop so file is valid and bounded in size
                if writer:
                    writer.release()
                    writer = None
            else:
                break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # --------------------------------------------------------
        # Multi-Object Tracking with ONNX Runtime & ByteTrack
        # --------------------------------------------------------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            verbose=False,
        )

        result = results[0]

        total_people = 0
        crowd_zone_people = 0
        restricted_violations = 0
        current_restricted_tids = set()

        # --------------------------------------------------------
        # Draw Zones and Line on Frame
        # --------------------------------------------------------
        draw_transparent_polygon(
            frame,
            crowd_poly,
            color=crowd_color,
            alpha=crowd_alpha,
            border_thickness=int(3 * scale),
        )

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

        draw_transparent_polygon(
            frame,
            restr_poly,
            color=restr_color,
            alpha=restr_alpha,
            border_thickness=int(3 * scale),
        )

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

        # Counting Line
        cv2.line(
            frame,
            line_start,
            line_end,
            line_color,
            int(4 * scale),
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

            for track_id, class_id, conf_val, box in zip(
                track_ids, class_ids, confidences, coordinates
            ):
                if class_id not in target_classes:
                    continue

                class_name = target_classes[class_id]
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                center = (cx, cy)

                if class_name == "person":
                    total_people += 1

                # Line Crossing
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
                        confidence=conf_val,
                    )

                # Crowd Zone
                in_crowd = False
                if class_name in crowd_target_classes:
                    if point_in_polygon(center, crowd_poly):
                        crowd_zone_people += 1
                        in_crowd = True

                # Restricted Zone
                in_restricted = False
                if class_name in restr_alert_classes:
                    if point_in_polygon(center, restr_poly):
                        restricted_violations += 1
                        in_restricted = True
                        current_restricted_tids.add(track_id)
                        logger.log_restricted_intrusion(
                            frame_idx=frame_idx,
                            object_class=class_name,
                            track_id=track_id,
                            zone_name=restr_cfg["name"],
                            x=cx,
                            y=cy,
                            confidence=conf_val,
                        )

                # Visual bounding box
                box_color = (0, 0, 255) if in_restricted else ((0, 255, 0) if in_crowd else (255, 180, 50))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, int(2 * scale))
                cv2.circle(frame, center, int(5 * scale), (0, 255, 255), -1)

                obj_label = f"{class_name} ID:{track_id} {conf_val:.2f}"
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

        # Overcrowding Logic
        overcrowding = crowd_zone_people >= crowd_threshold
        logger.log_crowd_status(
            frame_idx=frame_idx,
            count=crowd_zone_people,
            threshold=crowd_threshold,
            zone_name=crowd_cfg["name"],
        )

        if overcrowding:
            overcrowd_frames += 1
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
            intrusion_frames += restricted_violations

        # FPS & HUD
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

        if writer:
            writer.write(frame)

        if display:
            # Detect if user clicked the window's close button ('X')
            try:
                if cv2.getWindowProperty("Edge AI - Live Smart City Surveillance", cv2.WND_PROP_VISIBLE) < 1:
                    print("\nUser closed the surveillance window.")
                    break
            except Exception:
                pass

            # Scale frame for smooth GUI viewing on MacBook screen
            disp_view = cv2.resize(frame, (disp_w, disp_h))

            # If looping, display continuous demo status bar at bottom of window
            if loop and not is_webcam:
                cv2.putText(
                    disp_view,
                    f"CONTINUOUS LIVE DEMO | Loop {loop_count} | Press 'q' or close window to exit",
                    (15, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * (disp_w / 540.0),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Edge AI - Live Smart City Surveillance", disp_view)
            # 1ms delay for snappy live rendering
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\nUser requested exit (pressed 'q' or 'ESC').")
                break

        if frame_idx % 30 == 0:
            print(
                f"[Frame {frame_idx:03d}/{total_frames}] "
                f"FPS: {fps:4.1f} | "
                f"People: {total_people:2d} | "
                f"Crowd: {crowd_zone_people:2d}/{crowd_threshold} | "
                f"Restricted: {restricted_violations:2d} | "
                f"In/Out: {line_tracker.in_count}/{line_tracker.out_count}"
            )

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    logger.print_recent_events(limit=10)
    db_summary = logger.get_summary()
    print("Database Events Summary:")
    for evt, cnt in db_summary.items():
        print(f"  - {evt:<18}: {cnt}")
    logger.close()

    print("\n" + "=" * 70)
    print("EDGE INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Frames Processed : {frame_idx} across {loop_count} loop(s)")
    print(f"Crossings        : In {line_tracker.in_count} | Out {line_tracker.out_count}")
    print(f"Overcrowd Frames : {overcrowd_frames}")
    print(f"Intrusion Frames : {intrusion_frames}")
    if writer:
        print(f"Saved Video      : {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI Live Surveillance Engine")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model (defaults to yolov8n_int8.onnx)")
    parser.add_argument("--source", type=str, default=None, help="Video source (file path or '0' for webcam)")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video file")
    parser.add_argument("--no-loop", action="store_true", help="Disable continuous looping of video source")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
    args = parser.parse_args()

    run_edge_inference(
        model_path=args.model,
        source=args.source,
        display=not args.no_display,
        save_video=not args.no_save,
        conf_thresh=args.conf,
        max_frames=args.max_frames,
        loop=not args.no_loop,
    )
