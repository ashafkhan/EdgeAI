"""
Edge AI for Smart City Surveillance - Zone Calibration Utility
Allows visual selection of Crowd Zone, Restricted Zone, and Counting Line
coordinates directly on video frames and updates config/config.yaml.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import load_config


class ZoneCalibrator:
    def __init__(self, video_path: str, config_path: str = None):
        self.video_path = Path(video_path)
        self.config_path = (
            Path(config_path) if config_path else ROOT_DIR / "config" / "config.yaml"
        )
        self.cfg = load_config(self.config_path)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target display size for calibration window
        self.disp_h = 960
        self.scale_disp = self.disp_h / float(self.orig_h)
        self.disp_w = int(self.orig_w * self.scale_disp)

        # Load existing coordinates
        self.crowd_points = [
            tuple(pt) for pt in self.cfg["crowd_zone"].get("polygon", [])
        ]
        self.restr_points = [
            tuple(pt) for pt in self.cfg["restricted_zone"].get("polygon", [])
        ]
        self.line_points = [
            tuple(self.cfg["counting_line"].get("start", [0, 0])),
            tuple(self.cfg["counting_line"].get("end", [0, 0])),
        ]

        self.current_mode = "crowd"  # "crowd", "restricted", "line"
        self.frame_idx = 0
        self.current_frame = None

        print("\n" + "=" * 60)
        print("ZONE CALIBRATION UTILITY")
        print("=" * 60)
        print(f"Video : {self.video_path} ({self.orig_w}x{self.orig_h})")
        print("Modes: [1] Crowd Zone  [2] Restricted Zone  [3] Counting Line")
        print("Keys : [s] Save to config.yaml  [r] Reset current zone")
        print("       [n] Next frame           [q] Quit")
        print("=" * 60 + "\n")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coordinates back to original resolution
            orig_x = int(x / self.scale_disp)
            orig_y = int(y / self.scale_disp)

            if self.current_mode == "crowd":
                if len(self.crowd_points) >= 4:
                    self.crowd_points = []
                self.crowd_points.append((orig_x, orig_y))
                print(f"[Crowd Zone] Added Point {len(self.crowd_points)}: ({orig_x}, {orig_y})")

            elif self.current_mode == "restricted":
                if len(self.restr_points) >= 4:
                    self.restr_points = []
                self.restr_points.append((orig_x, orig_y))
                print(f"[Restricted Zone] Added Point {len(self.restr_points)}: ({orig_x}, {orig_y})")

            elif self.current_mode == "line":
                if len(self.line_points) >= 2:
                    self.line_points = []
                self.line_points.append((orig_x, orig_y))
                print(f"[Counting Line] Added Point {len(self.line_points)}: ({orig_x}, {orig_y})")

    def save_config(self):
        with open(self.config_path, "r") as f:
            cfg_data = yaml.safe_load(f)

        if len(self.crowd_points) >= 3:
            cfg_data["crowd_zone"]["polygon"] = [list(pt) for pt in self.crowd_points]
        if len(self.restr_points) >= 3:
            cfg_data["restricted_zone"]["polygon"] = [list(pt) for pt in self.restr_points]
        if len(self.line_points) == 2:
            cfg_data["counting_line"]["start"] = list(self.line_points[0])
            cfg_data["counting_line"]["end"] = list(self.line_points[1])

        with open(self.config_path, "w") as f:
            yaml.dump(cfg_data, f, default_flow_style=False, sort_keys=False)

        print(f"\n>>> Saved updated coordinates to: {self.config_path}\n")

    def run(self):
        cv2.namedWindow("Zone Calibrator", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Zone Calibrator", self.mouse_callback)

        ret, self.current_frame = self.cap.read()
        if not ret:
            print("Error: Could not read video frame.")
            return

        while True:
            display = self.current_frame.copy()

            # Draw Crowd Zone
            if len(self.crowd_points) >= 3:
                pts = np.array(self.crowd_points, dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 4)
            for p in self.crowd_points:
                cv2.circle(display, p, 8, (0, 255, 0), -1)

            # Draw Restricted Zone
            if len(self.restr_points) >= 3:
                pts = np.array(self.restr_points, dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 0, 255), 4)
            for p in self.restr_points:
                cv2.circle(display, p, 8, (0, 0, 255), -1)

            # Draw Counting Line
            if len(self.line_points) == 2:
                cv2.line(display, self.line_points[0], self.line_points[1], (0, 255, 255), 5)
            for p in self.line_points:
                cv2.circle(display, p, 8, (0, 255, 255), -1)

            # Resize to window display size
            scaled = cv2.resize(display, (self.disp_w, self.disp_h))

            # Draw UI Instructions on top of scaled frame
            mode_text = f"MODE: {self.current_mode.upper()} ([1] Crowd, [2] Restricted, [3] Line)"
            cv2.rectangle(scaled, (0, 0), (self.disp_w, 60), (20, 20, 20), -1)
            cv2.putText(scaled, mode_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(
                scaled,
                "Keys: [s] Save  [r] Reset current  [n] Next Frame  [q] Quit",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Zone Calibrator", scaled)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord("1"):
                self.current_mode = "crowd"
                print("Switched to Crowd Zone Mode")
            elif key == ord("2"):
                self.current_mode = "restricted"
                print("Switched to Restricted Zone Mode")
            elif key == ord("3"):
                self.current_mode = "line"
                print("Switched to Counting Line Mode")
            elif key == ord("r"):
                if self.current_mode == "crowd":
                    self.crowd_points = []
                elif self.current_mode == "restricted":
                    self.restr_points = []
                elif self.current_mode == "line":
                    self.line_points = []
                print(f"Reset {self.current_mode} points")
            elif key == ord("s"):
                self.save_config()
            elif key == ord("n"):
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.frame_idx += 1

        self.cap.release()
        cv2.destroyAllWindows()


def verify_config_zones():
    """Verify configuration coordinates non-interactively and render snapshot."""
    cfg = load_config()
    video_path = cfg["video"]["input_path"]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video frame for verification.")
        return False

    crowd_poly = np.array(cfg["crowd_zone"]["polygon"], dtype=np.int32)
    restr_poly = np.array(cfg["restricted_zone"]["polygon"], dtype=np.int32)
    l_start = tuple(cfg["counting_line"]["start"])
    l_end = tuple(cfg["counting_line"]["end"])

    overlay = frame.copy()
    cv2.fillPoly(overlay, [crowd_poly], (0, 200, 0))
    cv2.fillPoly(overlay, [restr_poly], (0, 0, 200))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    cv2.polylines(frame, [crowd_poly], True, (0, 255, 0), 4)
    cv2.polylines(frame, [restr_poly], True, (0, 0, 255), 4)
    cv2.line(frame, l_start, l_end, (0, 255, 255), 5)

    out_img = ROOT_DIR / "output" / "verified_zones.jpg"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    small = cv2.resize(frame, (540, 960))
    cv2.imwrite(str(out_img), small)
    print(f"Verification image saved to: {out_img}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zone Calibration Tool")
    parser.add_argument("--verify", action="store_true", help="Non-interactive zone verification")
    args = parser.parse_args()

    if args.verify:
        verify_config_zones()
    else:
        cfg = load_config()
        calibrator = ZoneCalibrator(cfg["video"]["input_path"])
        calibrator.run()
