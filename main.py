"""
main.py - Multi-Object Detection & Tracking Pipeline

End-to-end pipeline: detect → track → annotate → write video (with audio).

Audio handling
--------------
OpenCV cannot write audio. We write a silent annotated video first, then
use FFmpeg to mux the original audio track into the final output MP4.

Usage
-----
    python main.py --video <path_or_url> [OPTIONS]
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2

from detector.detector import Detector, DetectorConfig
from tracker.tracker import ObjectTracker, TrackerConfig, TrackerType
from draw.draw import DrawConfig, FrameAnnotator
from utils.video_io import (
    VideoReader, VideoWriter,
    download_video, is_url,
    mux_audio_into_video,
)
from utils.logger_utils import FPSCounter, TrackingLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-Object Detection & Tracking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--video", "-v", required=True,
                   help="Video file path OR public URL (YouTube / direct MP4)")
    p.add_argument("--output-dir", "-o", default="output",
                   help="Directory for annotated video and tracking logs")
    p.add_argument("--output-name", default=None,
                   help="Base name for output files (auto-derived if omitted)")
    p.add_argument("--display", action="store_true",
                   help="Show live annotated video while processing (needs GUI)")
    p.add_argument("--no-audio", action="store_true",
                   help="Skip audio muxing (output will be silent)")

    # Detection
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights: yolov8n/s/m/l/x.pt  or  yolo11*.pt")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Detection confidence threshold [0,1]")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--classes", nargs="+", default=None, metavar="CLASS",
                   help="Filter to class names, e.g. --classes person car")
    p.add_argument("--img-size", type=int, default=640,
                   help="YOLO inference image size")

    # Tracker
    p.add_argument("--tracker", choices=["bytetrack", "deepsort"],
                   default="bytetrack", help="Tracking algorithm")
    p.add_argument("--track-thresh", type=float, default=0.25,
                   help="ByteTrack low-score detection threshold")
    p.add_argument("--match-thresh", type=float, default=0.8,
                   help="ByteTrack IoU match threshold")
    p.add_argument("--track-buffer", type=int, default=30,
                   help="Frames to keep a lost track alive before deletion")

    # Visualization
    p.add_argument("--no-trail", action="store_true",
                   help="Disable motion trail visualization")
    p.add_argument("--trail-length", type=int, default=40,
                   help="Past positions to draw per trail")
    p.add_argument("--no-conf-label", action="store_true",
                   help="Hide confidence score on labels")

    # Processing
    p.add_argument("--skip", type=int, default=0,
                   help="Frame skip (0=every frame, 1=every other, …)")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Start at this frame index")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Stop at this frame index")
    p.add_argument("--device", default="auto",
                   help="Compute device: auto | cpu | cuda | mps")
    p.add_argument("--half", action="store_true",
                   help="FP16 inference (GPU only)")
    p.add_argument("--no-save-json", action="store_true",
                   help="Skip JSON log export (CSV always saved)")

    return p


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TrackingPipeline:
    """End-to-end detection + tracking + annotation + output pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        det_cfg = DetectorConfig(
            model_name=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            target_classes=args.classes,
            device=args.device,
            img_size=args.img_size,
            half_precision=args.half,
        )
        self.detector = Detector(det_cfg)

        trk_cfg = TrackerConfig(
            tracker_type=TrackerType(args.tracker),
            bt_track_thresh=args.track_thresh,
            bt_match_thresh=args.match_thresh,
            bt_track_buffer=args.track_buffer,
        )
        self.tracker = ObjectTracker(trk_cfg)

        draw_cfg = DrawConfig(
            show_trail=not args.no_trail,
            trail_length=args.trail_length,
            show_confidence=not args.no_conf_label,
        )
        self.annotator = FrameAnnotator(draw_cfg)

        self._output_dir = Path(args.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        args = self.args

        # ---- Resolve source ----
        video_path = args.video
        if is_url(video_path):
            logger.info("URL detected – downloading video …")
            video_path = download_video(
                video_path,
                output_dir=str(self._output_dir / "downloads"),
            )

        if not Path(video_path).is_file():
            logger.error("Video file not found: %s", video_path)
            sys.exit(1)

        # ---- Open reader ----
        reader = VideoReader(video_path)
        logger.info("Video: %s", reader)

        base_name = args.output_name or Path(video_path).stem
        final_out   = str(self._output_dir / f"{base_name}_tracked.mp4")
        # OpenCV writes to a temp file; FFmpeg muxes audio into final_out
        silent_tmp  = str(self._output_dir / f"{base_name}_silent_tmp.mp4")
        log_csv     = str(self._output_dir / "logs" / f"{base_name}_tracking.csv")
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)

        writer    = VideoWriter(silent_tmp, reader.fps, reader.width, reader.height)
        track_log = TrackingLogger(log_csv)
        fps_ctr   = FPSCounter(window=30)

        total = reader.total_frames
        logger.info(
            "Processing %d frames | skip=%d | device=%s | tracker=%s",
            total, args.skip, self.detector.device, args.tracker,
        )

        last_tracked = []

        try:
            for frame_id, frame in reader.frames(
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                skip=0,
            ):
                is_det_frame = (args.skip == 0) or (frame_id % (args.skip + 1) == 0)

                if is_det_frame:
                    detections   = self.detector.detect(frame)
                    tracked      = self.tracker.update(detections, frame)
                    last_tracked = tracked
                else:
                    tracked = last_tracked

                current_fps = fps_ctr.tick()

                annotated = self.annotator.annotate(
                    frame, tracked, fps=current_fps, frame_id=frame_id
                )
                self.annotator.clear_lost_tracks({o.track_id for o in tracked})

                writer.write(annotated)

                if is_det_frame:
                    track_log.log_frame(frame_id, tracked)

                if args.display:
                    cv2.imshow("Multi-Object Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User pressed Q – stopping.")
                        break

                if frame_id % 100 == 0:
                    pct = (frame_id / max(total, 1)) * 100
                    logger.info(
                        "  [%5.1f%%] frame %d/%d | FPS %.1f | objects %d",
                        pct, frame_id, total, current_fps, len(tracked),
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            reader.release()
            writer.release()
            if args.display:
                cv2.destroyAllWindows()

        # ---- Audio muxing ----
        audio_ok = False
        if not args.no_audio:
            start_secs = args.start_frame / max(reader.fps, 1.0)
            audio_ok = mux_audio_into_video(
                silent_video=silent_tmp,
                source_with_audio=video_path,
                output_path=final_out,
                start_time=start_secs,
            )
        if not audio_ok:
            # FFmpeg unavailable or failed — rename silent as final
            import shutil
            shutil.copy2(silent_tmp, final_out)

        # Clean up the silent temp file
        try:
            Path(silent_tmp).unlink(missing_ok=True)
        except Exception:
            pass

        # ---- Save logs ----
        if not args.no_save_json:
            track_log.save_json()
        track_log.close()
        summary = track_log.summary()

        # ---- Summary ----
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("  Annotated video  : %s  (audio=%s)", final_out, "yes" if audio_ok else "no")
        logger.info("  Tracking CSV     : %s", log_csv)
        logger.info("  Total detections : %d", summary.get("total_detections", 0))
        logger.info("  Unique tracks    : %d", summary.get("unique_tracks", 0))
        logger.info("  Frames processed : %d", summary.get("frames_processed", 0))
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Multi-Object Detection & Tracking Pipeline")
    logger.info("=" * 60)
    logger.info("  Video   : %s", args.video)
    logger.info("  Model   : %s", args.model)
    logger.info("  Tracker : %s", args.tracker)
    logger.info("  Conf    : %.2f", args.conf)
    logger.info("  Classes : %s", args.classes or "ALL")
    logger.info("  Device  : %s", args.device)
    logger.info("=" * 60)

    TrackingPipeline(args).run()


if __name__ == "__main__":
    main()
