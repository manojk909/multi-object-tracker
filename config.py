"""
config.py - Central configuration management

Provides dataclass-based config with sensible defaults.
Can be extended to support YAML/TOML config files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Input
    video_source: str = ""
    output_dir: str = "output"
    output_name: Optional[str] = None

    # Detection
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    target_classes: Optional[List[str]] = None
    img_size: int = 640
    device: str = "auto"
    half_precision: bool = False

    # Tracking
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.25
    match_thresh: float = 0.8
    track_buffer: int = 30

    # Visualization
    show_trail: bool = True
    trail_length: int = 40
    show_confidence: bool = True
    show_fps: bool = True
    display: bool = False

    # Processing
    frame_skip: int = 0
    start_frame: int = 0
    end_frame: Optional[int] = None

    # Output
    save_json: bool = True
