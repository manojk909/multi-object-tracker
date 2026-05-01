"""
tracker.py - Multi-Object Tracking Module

Supports two backends:
  - ByteTrack  (default, fast, no re-ID features required)
  - DeepSORT   (appearance-based re-ID, heavier)

Both produce TrackedObject instances with stable IDs across frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from detector.detector import Detection

logger = logging.getLogger(__name__)


class TrackerType(str, Enum):
    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"


@dataclass
class TrackedObject:
    """Single tracked object with stable ID."""

    track_id: int
    bbox_xyxy: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    age: int = 0                   # frames since first detection
    is_confirmed: bool = True

    def to_xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


@dataclass
class TrackerConfig:
    """Unified configuration for both tracker backends."""

    tracker_type: TrackerType = TrackerType.BYTETRACK

    # ---- ByteTrack knobs ----
    bt_track_thresh: float = 0.25       # low-score detection threshold
    bt_match_thresh: float = 0.8        # IoU match threshold
    bt_track_buffer: int = 30           # frames to keep lost track alive
    bt_frame_rate: int = 30             # nominal video FPS (for Kalman)

    # ---- DeepSORT knobs ----
    ds_max_age: int = 30                # max frames before track deletion
    ds_n_init: int = 3                  # frames before track is confirmed
    ds_max_cosine_distance: float = 0.4
    ds_nn_budget: int = 100             # max feature vectors per track
    ds_embedder: str = "mobilenet"      # "mobilenet" | "torchreid" | None


class ByteTrackWrapper:
    """
    Thin wrapper around supervision's ByteTrack implementation.

    supervision ≥ 0.21 ships ByteTrack natively; we also support
    a pure-Python fallback using the 'bytetracker' pip package.
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._tracker = None
        self._frame_count = 0
        self._load()

    def _load(self) -> None:
        # Primary: supervision ByteTrack
        try:
            import supervision as sv

            self._tracker = sv.ByteTracker(
                track_activation_threshold=self.cfg.bt_track_thresh,
                lost_track_buffer=self.cfg.bt_track_buffer,
                minimum_matching_threshold=self.cfg.bt_match_thresh,
                frame_rate=self.cfg.bt_frame_rate,
            )
            self._backend = "supervision"
            logger.info("ByteTrack backend: supervision")
            return
        except (ImportError, AttributeError):
            pass

        # Fallback: ultralytics built-in tracker config
        # We implement a minimal IoU-based tracker if neither is available
        logger.warning(
            "supervision ByteTrack not available – using built-in IoU tracker fallback"
        )
        self._tracker = _MinimalIoUTracker(
            max_age=self.cfg.bt_track_buffer,
            iou_threshold=self.cfg.bt_match_thresh,
        )
        self._backend = "iou_fallback"

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedObject]:
        self._frame_count += 1

        if not detections:
            if self._backend == "supervision":
                import supervision as sv
                empty = sv.Detections.empty()
                self._tracker.update(detections=empty)
            elif self._backend == "iou_fallback":
                self._tracker.update([])
            return []

        if self._backend == "supervision":
            return self._update_supervision(detections, frame)
        return self._update_iou_fallback(detections)

    def _update_supervision(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedObject]:
        import supervision as sv

        xyxy = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        confs = np.array([d.confidence for d in detections], dtype=np.float32)
        cls_ids = np.array([d.class_id for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=cls_ids,
        )
        tracked = self._tracker.update(detections=sv_dets)

        results: List[TrackedObject] = []
        if tracked is None or len(tracked) == 0:
            return results

        for i in range(len(tracked)):
            tid = (
                int(tracked.tracker_id[i])
                if tracked.tracker_id is not None
                else -1
            )
            if tid < 0:
                continue
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            # Map back to class name
            cls_name = detections[0].class_name if detections else str(cls_id)
            # Find best matching original detection for class name
            for d in detections:
                if d.class_id == cls_id:
                    cls_name = d.class_name
                    break

            results.append(
                TrackedObject(
                    track_id=tid,
                    bbox_xyxy=tracked.xyxy[i].astype(np.float32),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )
        return results

    def _update_iou_fallback(
        self, detections: List[Detection]
    ) -> List[TrackedObject]:
        raw = [
            {
                "bbox": d.bbox_xyxy,
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
            }
            for d in detections
        ]
        tracked_raw = self._tracker.update(raw)
        return [
            TrackedObject(
                track_id=t["track_id"],
                bbox_xyxy=t["bbox"],
                confidence=t["confidence"],
                class_id=t["class_id"],
                class_name=t["class_name"],
            )
            for t in tracked_raw
        ]


class DeepSORTWrapper:
    """Wrapper around deep_sort_realtime."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._tracker = None
        self._class_map: Dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._tracker = DeepSort(
                max_age=self.cfg.ds_max_age,
                n_init=self.cfg.ds_n_init,
                max_cosine_distance=self.cfg.ds_max_cosine_distance,
                nn_budget=self.cfg.ds_nn_budget,
                embedder=self.cfg.ds_embedder,
                half=False,
                bgr=True,
            )
            logger.info(f"DeepSORT loaded (embedder={self.cfg.ds_embedder})")
        except ImportError as exc:
            raise RuntimeError(
                "deep_sort_realtime not installed. Run: pip install deep-sort-realtime"
            ) from exc

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedObject]:
        if not detections:
            # still update to age-out lost tracks
            self._tracker.update_tracks([], frame=frame)
            return []

        # DeepSORT expects [[x1,y1,w,h], conf, cls_id]
        raw = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            raw.append(([x1, y1, x2 - x1, y2 - y1], d.confidence, d.class_id))
            self._class_map[d.class_id] = d.class_name

        tracks = self._tracker.update_tracks(raw, frame=frame)
        results: List[TrackedObject] = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            cls_id = track.det_class if track.det_class is not None else -1
            results.append(
                TrackedObject(
                    track_id=int(track.track_id),
                    bbox_xyxy=np.array(ltrb, dtype=np.float32),
                    confidence=track.det_conf if track.det_conf is not None else 0.0,
                    class_id=cls_id,
                    class_name=self._class_map.get(cls_id, str(cls_id)),
                    age=track.age,
                    is_confirmed=track.is_confirmed(),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Minimal IoU tracker (zero-dependency fallback)
# ---------------------------------------------------------------------------

class _MinimalIoUTracker:
    """
    Simple IoU-based multi-object tracker.
    Used when neither supervision nor deep_sort_realtime is available.
    Implements greedy IoU matching with Kalman-less track aging.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: List[dict]) -> List[dict]:
        # Age existing tracks
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        if not detections:
            return []

        if not self._tracks:
            # Init all as new tracks
            for d in detections:
                self._tracks[self._next_id] = {
                    "bbox": d["bbox"],
                    "confidence": d["confidence"],
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "age": 0,
                }
                self._next_id += 1
        else:
            track_ids = list(self._tracks.keys())
            track_boxes = np.array([self._tracks[t]["bbox"] for t in track_ids])
            det_boxes = np.array([d["bbox"] for d in detections])

            iou_mat = _batch_iou(track_boxes, det_boxes)
            matched_tracks = set()
            matched_dets = set()

            # Greedy match
            for _ in range(min(len(track_ids), len(detections))):
                if iou_mat.size == 0:
                    break
                r, c = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[r, c] < self.iou_threshold:
                    break
                tid = track_ids[r]
                d = detections[c]
                self._tracks[tid].update(
                    bbox=d["bbox"],
                    confidence=d["confidence"],
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    age=0,
                )
                matched_tracks.add(r)
                matched_dets.add(c)
                iou_mat[r, :] = -1
                iou_mat[:, c] = -1

            # New tracks for unmatched detections
            for ci, d in enumerate(detections):
                if ci not in matched_dets:
                    self._tracks[self._next_id] = {
                        "bbox": d["bbox"],
                        "confidence": d["confidence"],
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                        "age": 0,
                    }
                    self._next_id += 1

        return [
            {
                "track_id": tid,
                "bbox": info["bbox"],
                "confidence": info["confidence"],
                "class_id": info["class_id"],
                "class_name": info["class_name"],
            }
            for tid, info in self._tracks.items()
        ]


def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of [x1,y1,x2,y2] boxes."""
    # boxes_a: (M, 4), boxes_b: (N, 4) → (M, N)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    Unified tracker interface. Choose backend via TrackerConfig.

    Usage
    -----
    tracker = ObjectTracker(config)
    tracked = tracker.update(detections, frame)
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        if config.tracker_type == TrackerType.BYTETRACK:
            self._backend = ByteTrackWrapper(config)
        elif config.tracker_type == TrackerType.DEEPSORT:
            self._backend = DeepSORTWrapper(config)
        else:
            raise ValueError(f"Unknown tracker type: {config.tracker_type}")

        logger.info(f"ObjectTracker initialized: {config.tracker_type.value}")

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedObject]:
        """Update tracker with new detections and return tracked objects."""
        return self._backend.update(detections, frame)

    def reset(self) -> None:
        """Reset tracker state (e.g., between video segments)."""
        self._backend = (
            ByteTrackWrapper(self.config)
            if self.config.tracker_type == TrackerType.BYTETRACK
            else DeepSORTWrapper(self.config)
        )
