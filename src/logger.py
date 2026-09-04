"""
Edge AI for Smart City Surveillance - Phase 5 Event Logger
Handles persistent event logging to SQLite database and CSV export.
Includes deduplication/debouncing for edge analytics events.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sqlite3
import csv

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "output" / "logs" / "surveillance.db"
DEFAULT_CSV_PATH = ROOT_DIR / "output" / "logs" / "events.csv"


class SurveillanceLogger:
    """
    Thread-safe SQLite & CSV event logger for edge surveillance.
    Supports debouncing to prevent database spamming.
    """

    def __init__(self, db_path: str = None, csv_path: str = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

        # State tracking for debouncing / deduplication
        self.active_intrusions: set[int] = set()  # track_ids currently inside restricted zone
        self.is_overcrowded: bool = False         # state of crowd zone
        self.overcrowd_log_cooldown: int = 0      # frames cooldown between repeated crowd logs

    def _init_db(self):
        """Initialize SQLite tables and indexes."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                frame_idx INTEGER,
                object_class TEXT,
                track_id INTEGER,
                event_type TEXT NOT NULL,
                zone TEXT,
                confidence REAL,
                x INTEGER,
                y INTEGER,
                details TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
        """)
        self.conn.commit()

    def log_event(
        self,
        event_type: str,
        frame_idx: int = None,
        object_class: str = None,
        track_id: int = None,
        zone: str = None,
        confidence: float = None,
        x: int = None,
        y: int = None,
        details: str = None,
    ):
        """
        Insert an event record into SQLite and append to CSV.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (
                timestamp, frame_idx, object_class, track_id,
                event_type, zone, confidence, x, y, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                frame_idx,
                object_class,
                track_id,
                event_type,
                zone,
                confidence,
                x,
                y,
                details,
            ),
        )
        self.conn.commit()

        # Append to CSV
        file_exists = self.csv_path.is_file() and self.csv_path.stat().st_size > 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "id", "timestamp", "frame_idx", "object_class",
                    "track_id", "event_type", "zone", "confidence",
                    "x", "y", "details"
                ])
            writer.writerow([
                cursor.lastrowid, now_str, frame_idx, object_class,
                track_id, event_type, zone, confidence,
                x, y, details
            ])

    def log_line_crossing(
        self,
        frame_idx: int,
        object_class: str,
        track_id: int,
        direction: str,
        x: int,
        y: int,
        confidence: float,
    ):
        """Log a virtual line crossing event (enter / exit)."""
        event_type = "enter" if "In" in direction or "East" in direction else "exit"
        self.log_event(
            event_type=event_type,
            frame_idx=frame_idx,
            object_class=object_class,
            track_id=track_id,
            zone="counting_line",
            confidence=confidence,
            x=x,
            y=y,
            details=f"Crossing: {direction}",
        )

    def log_restricted_intrusion(
        self,
        frame_idx: int,
        object_class: str,
        track_id: int,
        zone_name: str,
        x: int,
        y: int,
        confidence: float,
    ):
        """
        Log restricted zone intrusion with debounce:
        Logs only once when an object enters, until it leaves.
        """
        if track_id not in self.active_intrusions:
            self.active_intrusions.add(track_id)
            self.log_event(
                event_type="restricted_zone",
                frame_idx=frame_idx,
                object_class=object_class,
                track_id=track_id,
                zone=zone_name,
                confidence=confidence,
                x=x,
                y=y,
                details=f"Unauthorized entry by {object_class} ID:{track_id}",
            )

    def update_restricted_tracks(self, current_restricted_tids: set[int]):
        """Clear tracks that have exited the restricted zone so they can re-trigger if re-entering."""
        self.active_intrusions = self.active_intrusions.intersection(current_restricted_tids)

    def log_crowd_status(
        self,
        frame_idx: int,
        count: int,
        threshold: int,
        zone_name: str,
    ):
        """
        Log overcrowding events when threshold is breached or periodic alerts.
        """
        if count >= threshold:
            if not self.is_overcrowded or self.overcrowd_log_cooldown <= 0:
                self.is_overcrowded = True
                self.overcrowd_log_cooldown = 45  # Cooldown frames (~1.5s at 30fps)
                self.log_event(
                    event_type="overcrowding",
                    frame_idx=frame_idx,
                    object_class="person",
                    zone=zone_name,
                    details=f"Overcrowding alert: {count} people (limit {threshold})",
                )
            else:
                self.overcrowd_log_cooldown -= 1
        else:
            if self.is_overcrowded:
                self.is_overcrowded = False
                self.overcrowd_log_cooldown = 0
                self.log_event(
                    event_type="crowd_cleared",
                    frame_idx=frame_idx,
                    object_class="person",
                    zone=zone_name,
                    details=f"Crowd normalized: {count} people (limit {threshold})",
                )

    def get_summary(self) -> dict:
        """Query summary statistics from database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT event_type, COUNT(*) FROM events GROUP BY event_type
        """)
        summary = dict(cursor.fetchall())
        cursor.execute("SELECT COUNT(*) FROM events")
        summary["total_events"] = cursor.fetchone()[0]
        return summary

    def print_recent_events(self, limit: int = 15):
        """Print recent events in a tabular format."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, frame_idx, event_type, object_class, track_id, zone, details
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        print("\n" + "=" * 80)
        print(f"SURVEILLANCE EVENT LOG (Recent {len(rows)} Records)")
        print("=" * 80)
        print(f"{'ID':<5} | {'Timestamp':<19} | {'Frame':<5} | {'Event':<15} | {'Class':<8} | {'Track':<5} | {'Details'}")
        print("-" * 80)
        for r in reversed(rows):
            tid = str(r[5]) if r[5] is not None else "-"
            cls = r[4] if r[4] is not None else "-"
            print(f"{r[0]:<5} | {r[1]:<19} | {r[2] or 0:<5} | {r[3]:<15} | {cls:<8} | {tid:<5} | {r[7]}")
        print("=" * 80 + "\n")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surveillance Event Logger & Database CLI")
    parser.add_argument("--summary", action="store_true", help="Print event count summary")
    parser.add_argument("--recent", type=int, default=15, help="Print recent N events")
    args = parser.parse_args()

    logger = SurveillanceLogger()
    if args.summary:
        summary = logger.get_summary()
        print("Event Summary in SQLite Database:")
        for k, v in summary.items():
            print(f"  {k:<18}: {v}")
    logger.print_recent_events(limit=args.recent)
    logger.close()
