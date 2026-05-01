"""
draw.py - Annotation and Visualization Module

Renders bounding boxes, ID labels, class names, confidence scores,
motion trails, and a live FPS counter onto video frames.
"""

from __future__ import annotations

import colorsys
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracker.tracker import TrackedObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette helpers
# ---------------------------------------------------------------------------

def _generate_color_palette(n: int = 256) -> List[Tuple[int, int, int]]:
    """
    Generate perceptually distinct BGR colors using golden-ratio hue stepping.
    Returns a list of (B, G, R) tuples.
    """
    palette = []
    golden_ratio = 0.618033988749895
    hue = 0.0
    for _ in range(n):
        hue = (hue + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    return palette


_COLOR_PALETTE = _generate_color_palette(512)


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR color for a given track ID."""
    return _COLOR_PALETTE[track_id % len(_COLOR_PALETTE)]


# ---------------------------------------------------------------------------
# Drawing configuration
# ---------------------------------------------------------------------------

class DrawConfig:
    """Visual parameters for annotation rendering."""

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        font_thickness: int = 1,
        show_confidence: bool = True,
        show_class: bool = True,
        show_trail: bool = True,
        trail_length: int = 40,
        trail_thickness: int = 2,
        show_fps: bool = True,
        label_bg_alpha: float = 0.65,
    ) -> None:
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.show_confidence = show_confidence
        self.show_class = show_class
        self.show_trail = show_trail
        self.trail_length = trail_length
        self.trail_thickness = trail_thickness
        self.show_fps = show_fps
        self.label_bg_alpha = label_bg_alpha
        self.font = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Frame annotator
# ---------------------------------------------------------------------------

class FrameAnnotator:
    """
    Stateful annotator that maintains per-track motion trails.

    Usage
    -----
    annotator = FrameAnnotator(config)
    annotated_frame = annotator.annotate(frame, tracked_objects, fps=30.0)
    """

    def __init__(self, config: Optional[DrawConfig] = None) -> None:
        self.config = config or DrawConfig()
        # trail_history[track_id] = deque of (cx, cy) center points
        self._trail_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.config.trail_length)
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        fps: Optional[float] = None,
        frame_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw all annotations onto a copy of the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV.
        tracked_objects : List[TrackedObject]
            Tracked objects to render.
        fps : float, optional
            Current FPS to display in corner.
        frame_id : int, optional
            Frame number for corner overlay.

        Returns
        -------
        np.ndarray
            Annotated frame (same size as input).
        """
        output = frame.copy()

        # Update trail history
        for obj in tracked_objects:
            cx, cy = obj.center()
            self._trail_history[obj.track_id].append((cx, cy))

        # Draw trails FIRST (under boxes)
        if self.config.show_trail:
            self._draw_trails(output, tracked_objects)

        # Draw boxes and labels
        for obj in tracked_objects:
            self._draw_object(output, obj)

        # HUD overlay
        self._draw_hud(output, fps=fps, frame_id=frame_id, n_objects=len(tracked_objects))

        return output

    def clear_lost_tracks(self, active_ids: set) -> None:
        """Remove trail history for tracks that are no longer active."""
        stale = [tid for tid in self._trail_history if tid not in active_ids]
        for tid in stale:
            del self._trail_history[tid]

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _draw_object(self, frame: np.ndarray, obj: TrackedObject) -> None:
        cfg = self.config
        color = get_track_color(obj.track_id)
        x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.box_thickness)

        # Corner accent marks
        self._draw_corner_marks(frame, x1, y1, x2, y2, color, thickness=cfg.box_thickness + 1, length=12)

        # Label text
        parts = [f"#{obj.track_id}"]
        if cfg.show_class:
            parts.append(obj.class_name)
        if cfg.show_confidence:
            parts.append(f"{obj.confidence:.2f}")
        label = " | ".join(parts)

        self._draw_label(frame, label, x1, y1, color)

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        cfg = self.config
        (tw, th), baseline = cv2.getTextSize(label, cfg.font, cfg.font_scale, cfg.font_thickness)
        pad = 4

        # Position label above box; clamp to frame top
        label_y1 = max(y - th - baseline - pad * 2, 0)
        label_y2 = label_y1 + th + baseline + pad * 2

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, label_y1), (x + tw + pad * 2, label_y2), color, -1)
        cv2.addWeighted(overlay, cfg.label_bg_alpha, frame, 1 - cfg.label_bg_alpha, 0, frame)

        # White text
        cv2.putText(
            frame,
            label,
            (x + pad, label_y2 - baseline - pad),
            cfg.font,
            cfg.font_scale,
            (255, 255, 255),
            cfg.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_trails(
        self, frame: np.ndarray, tracked_objects: List[TrackedObject]
    ) -> None:
        cfg = self.config
        for obj in tracked_objects:
            history = list(self._trail_history[obj.track_id])
            if len(history) < 2:
                continue
            color = get_track_color(obj.track_id)
            n = len(history)
            for i in range(1, n):
                # Fade older trail points (alpha proportional to age)
                alpha = i / n
                t_color = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(cfg.trail_thickness * alpha))
                cv2.line(frame, history[i - 1], history[i], t_color, thickness, cv2.LINE_AA)

    def _draw_corner_marks(
        self,
        frame: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int,
        color: Tuple[int, int, int],
        thickness: int = 3,
        length: int = 14,
    ) -> None:
        """Draw L-shaped corner accents inside the bounding box corners."""
        corners = [
            # (start_h, start_v, dx_h, dy_h, dx_v, dy_v)
            (x1, y1, length, 0, 0, length),
            (x2, y1, -length, 0, 0, length),
            (x1, y2, length, 0, 0, -length),
            (x2, y2, -length, 0, 0, -length),
        ]
        for cx, cy, dx1, dy1, dx2, dy2 in corners:
            cv2.line(frame, (cx, cy), (cx + dx1, cy + dy1), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx + dx2, cy + dy2), color, thickness, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: Optional[float],
        frame_id: Optional[int],
        n_objects: int,
    ) -> None:
        cfg = self.config
        h, w = frame.shape[:2]
        lines = []

        if fps is not None and cfg.show_fps:
            lines.append(f"FPS: {fps:.1f}")
        if frame_id is not None:
            lines.append(f"Frame: {frame_id}")
        lines.append(f"Objects: {n_objects}")

        x_start, y_start = 10, 28
        line_height = 22

        # Dark background panel
        panel_w = 140
        panel_h = len(lines) * line_height + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            y = y_start + i * line_height
            cv2.putText(
                frame,
                line,
                (x_start, y),
                cfg.font,
                0.52,
                (0, 230, 120),
                1,
                cv2.LINE_AA,
            )

        # Thin green border accent
        cv2.rectangle(frame, (5, 5), (5 + panel_w, 5 + panel_h), (0, 200, 80), 1)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake_objects = [
        TrackedObject(
            track_id=1,
            bbox_xyxy=np.array([100, 100, 300, 400], dtype=np.float32),
            confidence=0.91,
            class_id=0,
            class_name="person",
        ),
        TrackedObject(
            track_id=7,
            bbox_xyxy=np.array([500, 200, 750, 600], dtype=np.float32),
            confidence=0.76,
            class_id=0,
            class_name="person",
        ),
    ]
    annotator = FrameAnnotator()
    result = annotator.annotate(blank, fake_objects, fps=29.7, frame_id=42)
    cv2.imwrite("/tmp/draw_test.jpg", result)
    print("Saved /tmp/draw_test.jpg")
