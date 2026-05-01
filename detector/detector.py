"""
detector.py - YOLO-based Object Detector Module

Wraps Ultralytics YOLOv8/YOLOv11 for configurable, class-filtered
multi-object detection returning normalized detection arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# COCO class name → class ID mapping (subset shown; full list in ultralytics)
COCO_CLASSES: Dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "sports ball": 32,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "horse": 17,
    "dog": 16,
    "cat": 15,
}


@dataclass
class Detection:
    """Single detection result from the detector."""

    bbox_xyxy: np.ndarray          # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int
    class_name: str

    def to_xywh(self) -> np.ndarray:
        """Convert [x1,y1,x2,y2] → [x,y,w,h] (top-left + size)."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectorConfig:
    """Configuration for the YOLO detector."""

    model_name: str = "yolov8n.pt"          # yolov8n/s/m/l/x or yolo11*
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45             # NMS IoU threshold
    target_classes: Optional[List[str]] = None  # None = all classes
    device: str = "auto"                    # "auto", "cpu", "cuda", "mps"
    img_size: int = 640
    half_precision: bool = False            # FP16 – GPU only
    max_detections: int = 300

    # Derived: resolved class IDs for filtering (populated in Detector.__post_init__)
    _target_class_ids: List[int] = field(default_factory=list, init=False, repr=False)


class Detector:
    """
    Wraps Ultralytics YOLO for multi-class object detection.

    Usage
    -----
    detector = Detector(config)
    detections = detector.detect(frame_bgr)
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._device = self._resolve_device(config.device)
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (H, W, 3).

        Returns
        -------
        List[Detection]
            Filtered and sorted (by confidence desc) detections.
        """
        if frame is None or frame.size == 0:
            return []

        results = self._model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            max_det=self.config.max_detections,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)

                # Class filtering
                if self.config._target_class_ids and cls_id not in self.config._target_class_ids:
                    continue

                cls_name = self._class_names.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        bbox_xyxy=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    @property
    def device(self) -> str:
        return str(self._device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download / load YOLO model and resolve class IDs."""
        try:
            from ultralytics import YOLO  # lazy import

            logger.info(f"Loading YOLO model: {self.config.model_name} on {self._device}")
            self._model = YOLO(self.config.model_name)
            self._model.to(self._device)

            if self.config.half_precision and self._device != "cpu":
                self._model.model.half()
                logger.info("Half-precision (FP16) enabled")

            # Build class name map from model metadata
            if hasattr(self._model, "names"):
                self._class_names = {int(k): v for k, v in self._model.names.items()}
            else:
                self._class_names = {}

            # Resolve target class IDs
            if self.config.target_classes:
                resolved = []
                name_to_id = {v.lower(): k for k, v in self._class_names.items()}
                for cls_name in self.config.target_classes:
                    cls_id = name_to_id.get(cls_name.lower())
                    if cls_id is not None:
                        resolved.append(cls_id)
                        logger.info(f"  Tracking class: '{cls_name}' (id={cls_id})")
                    else:
                        logger.warning(f"  Class '{cls_name}' not found in model – skipping")
                self.config._target_class_ids = resolved
            else:
                logger.info("  Tracking ALL classes")

        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics not installed. Run: pip install ultralytics"
            ) from exc

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import cv2

    logging.basicConfig(level=logging.INFO)
    cfg = DetectorConfig(
        model_name="yolov8n.pt",
        confidence_threshold=0.4,
        target_classes=["person"],
    )
    det = Detector(cfg)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        dets = det.detect(frame)
        print(f"Detected {len(dets)} objects:")
        for d in dets:
            print(f"  {d.class_name} ({d.confidence:.2f}) @ {d.bbox_xyxy}")
    cap.release()
