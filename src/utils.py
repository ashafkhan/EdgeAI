"""
Edge AI for Smart City Surveillance - Utility Module
Provides configuration parsing, geometric algorithms, line crossing tracking,
and surveillance HUD visualization.
"""

from pathlib import Path
import cv2
import numpy as np
import yaml

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file and resolve relative paths.
    """
    if config_path is None:
        config_path = ROOT_DIR / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to ROOT_DIR
    if "model" in cfg and "path" in cfg["model"]:
        cfg["model"]["path"] = str(ROOT_DIR / cfg["model"]["path"])
    if "video" in cfg:
        if "input_path" in cfg["video"]:
            cfg["video"]["input_path"] = str(ROOT_DIR / cfg["video"]["input_path"])
        if "output_dir" in cfg["video"]:
            cfg["video"]["output_dir"] = str(ROOT_DIR / cfg["video"]["output_dir"])

    return cfg


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Check whether a 2D point (x, y) is inside or on the boundary of a polygon.
    """
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0


def get_line_side(point: tuple[int, int], line_start: tuple[int, int], line_end: tuple[int, int]) -> float:
    """
    Determine which side of an oriented 2D line segment the point is on.
    Returns:
        > 0 : Left side (Positive side)
        < 0 : Right side (Negative side)
        = 0 : Collinear
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class LineCrossingTracker:
    """
    Robust virtual line crossing detector using persistent track IDs.
    
    Prevents duplicate counts when objects linger near the line,
    while permitting legitimate future re-entry / exit events.
    """

    def __init__(
        self,
        line_start: tuple[int, int],
        line_end: tuple[int, int],
        target_classes: list[str] = None,
        padding: int = 80,
    ):
        self.line_start = tuple(map(int, line_start))
        self.line_end = tuple(map(int, line_end))
        self.target_classes = target_classes or ["person"]
        self.padding = padding

        # Track state
        self.prev_sides: dict[int, float] = {}
        self.last_crossed: dict[int, str] = {}  # track_id -> 'pos' or 'neg'

        # Counters
        self.in_count: int = 0
        self.out_count: int = 0

        # Precompute bounding box of the line segment with padding
        x_min = min(self.line_start[0], self.line_end[0]) - self.padding
        x_max = max(self.line_start[0], self.line_end[0]) + self.padding
        y_min = min(self.line_start[1], self.line_end[1]) - self.padding
        y_max = max(self.line_start[1], self.line_end[1]) + self.padding
        self.bbox = (x_min, y_min, x_max, y_max)

    def is_near_line(self, point: tuple[int, int]) -> bool:
        """Check if a point is within the bounding span of the line segment."""
        x, y = point
        return (
            self.bbox[0] <= x <= self.bbox[2]
            and self.bbox[1] <= y <= self.bbox[3]
        )

    def update(self, track_id: int, center: tuple[int, int], class_name: str) -> str | None:
        """
        Update tracker with current object position.
        Returns:
            'in' if crossed negative -> positive
            'out' if crossed positive -> negative
            None otherwise
        """
        if self.target_classes and class_name not in self.target_classes:
            return None

        current_side = get_line_side(center, self.line_start, self.line_end)
        event = None

        if track_id in self.prev_sides and self.is_near_line(center):
            prev_side = self.prev_sides[track_id]
            last_event = self.last_crossed.get(track_id)

            # Crossing negative -> positive (Entry / In)
            if prev_side < 0 and current_side >= 0:
                if last_event != "pos":
                    self.in_count += 1
                    self.last_crossed[track_id] = "pos"
                    event = "in"

            # Crossing positive -> negative (Exit / Out)
            elif prev_side > 0 and current_side <= 0:
                if last_event != "neg":
                    self.out_count += 1
                    self.last_crossed[track_id] = "neg"
                    event = "out"

        self.prev_sides[track_id] = current_side
        return event


def draw_transparent_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.25,
    border_thickness: int = 3,
) -> None:
    """
    Draw a polygon on frame with a semi-transparent colored fill and solid border.
    Modifies frame in-place.
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=border_thickness)


def draw_text_box(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.8,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 6,
) -> tuple[int, int]:
    """
    Draw text with a solid rectangular background for high visibility.
    Returns (box_width, box_height).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin

    # Background rectangle
    pt1 = (x, y - text_h - padding)
    pt2 = (x + text_w + padding * 2, y + baseline + padding)
    cv2.rectangle(frame, pt1, pt2, bg_color, -1)

    # Text
    cv2.putText(
        frame,
        text,
        (x + padding, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return text_w + padding * 2, text_h + baseline + padding * 2


def draw_hud(
    frame: np.ndarray,
    fps: float,
    total_people: int,
    crowd_zone_people: int,
    crowd_threshold: int,
    overcrowding: bool,
    restricted_violations: int,
    entry_count: int,
    exit_count: int,
    scale_factor: float = 1.0,
) -> None:
    """
    Draw a sleek, professional surveillance analytics HUD overlay.
    Dynamically scales according to video resolution.
    """
    h, w = frame.shape[:2]

    # Dynamically compute font scale based on frame width (baseline 1920)
    base_scale = (w / 1920.0) * 0.75 * scale_factor
    f_scale = max(0.6, base_scale)
    line_spacing = int(42 * (f_scale / 0.75))
    hud_x = int(30 * (w / 1920.0))
    hud_y = int(50 * (h / 1080.0))

    # Semi-transparent HUD background panel
    panel_w = int(540 * (w / 1920.0))
    panel_h = line_spacing * 8 + int(30 * (h / 1080.0))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (hud_x - 15, hud_y - 35),
        (hud_x + panel_w, hud_y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(
        frame,
        (hud_x - 15, hud_y - 35),
        (hud_x + panel_w, hud_y + panel_h),
        (80, 80, 80),
        2,
    )

    # Header
    cv2.putText(
        frame,
        "EDGE AI SURVEILLANCE ENGINE",
        (hud_x, hud_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale * 1.05,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (hud_x, hud_y + 10),
        (hud_x + panel_w - 30, hud_y + 10),
        (0, 255, 255),
        2,
    )

    curr_y = hud_y + line_spacing + 5

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (hud_x, curr_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    curr_y += line_spacing

    # Total People Detected
    cv2.putText(
        frame,
        f"Frame People: {total_people}",
        (hud_x, curr_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    curr_y += line_spacing

    # Crowd Zone People
    crowd_color = (0, 0, 255) if overcrowding else (0, 255, 0)
    cv2.putText(
        frame,
        f"Crowd Zone: {crowd_zone_people} / {crowd_threshold} limit",
        (hud_x, curr_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale,
        crowd_color,
        2,
        cv2.LINE_AA,
    )
    curr_y += line_spacing

    # Overcrowding status banner
    status_text = "STATUS: [OVERCROWDED]" if overcrowding else "STATUS: NORMAL"
    status_color = (0, 0, 255) if overcrowding else (0, 220, 0)
    cv2.putText(
        frame,
        status_text,
        (hud_x, curr_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale * 1.05,
        status_color,
        2,
        cv2.LINE_AA,
    )
    curr_y += line_spacing

    # Restricted Zone Violations
    restr_color = (0, 0, 255) if restricted_violations > 0 else (180, 180, 180)
    cv2.putText(
        frame,
        f"Restricted Violations: {restricted_violations}",
        (hud_x, curr_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale,
        restr_color,
        2,
        cv2.LINE_AA,
    )
    curr_y += line_spacing

    # Line crossing counts
    cv2.putText(
        frame,
        f"Crossings In: {entry_count} | Out: {exit_count}",
        (hud_x, curr_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_scale,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
