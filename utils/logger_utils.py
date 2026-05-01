"""
logger_utils.py - Tracking Log Export Utilities

Writes per-frame tracking results to CSV and JSON formats,
and provides a simple FPS counter utility.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracker.tracker import TrackedObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class TrackingLogger:
    """
    Logs tracking results to CSV and optionally JSON.

    CSV schema
    ----------
    frame_id, track_id, class_id, class_name, x1, y1, x2, y2, confidence
    """

    CSV_FIELDS = [
        "frame_id", "track_id", "class_id", "class_name",
        "x1", "y1", "x2", "y2", "confidence",
    ]

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []
        self._csv_file = None
        self._csv_writer = None
        self._open_csv()

    def _open_csv(self) -> None:
        self._csv_file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self.CSV_FIELDS
        )
        self._csv_writer.writeheader()

    def log_frame(
        self, frame_id: int, tracked_objects: List[TrackedObject]
    ) -> None:
        """Append tracking results for a single frame."""
        for obj in tracked_objects:
            x1, y1, x2, y2 = [round(float(v), 2) for v in obj.bbox_xyxy]
            row = {
                "frame_id": frame_id,
                "track_id": obj.track_id,
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": round(obj.confidence, 4),
            }
            self._rows.append(row)
            self._csv_writer.writerow(row)

    def save_json(self, json_path: Optional[str] = None) -> str:
        """Also dump all logs as JSON (grouped by frame)."""
        if json_path is None:
            json_path = str(self.output_path.with_suffix(".json"))

        grouped: Dict[int, List[Dict]] = {}
        for row in self._rows:
            fid = row["frame_id"]
            grouped.setdefault(fid, []).append(
                {k: v for k, v in row.items() if k != "frame_id"}
            )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2)

        logger.info(f"Tracking log (JSON): {json_path}")
        return json_path

    def close(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        logger.info(
            f"Tracking log (CSV): {self.output_path} | "
            f"{len(self._rows)} total detections"
        )

    def summary(self) -> Dict[str, Any]:
        """Return basic statistics about the logged session."""
        if not self._rows:
            return {}
        track_ids = {r["track_id"] for r in self._rows}
        frame_ids = {r["frame_id"] for r in self._rows}
        return {
            "total_detections": len(self._rows),
            "unique_tracks": len(track_ids),
            "frames_processed": len(frame_ids),
        }

    def __enter__(self) -> "TrackingLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
