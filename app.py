"""
app.py - Gradio Web Interface for Multi-Object Tracking Pipeline

Provides a browser-based UI for uploading/linking videos and
running the full detection + tracking pipeline.

Deploy to Hugging Face Spaces (free):
    1. Push this repo to GitHub
    2. Create a new Space at https://huggingface.co/new-space
       - SDK: Gradio
       - Link your GitHub repo
    3. HF Spaces auto-builds and hosts the app

Run locally:
    pip install gradio
    python app.py
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

# ── ensure project root is importable ─────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from detector.detector import Detector, DetectorConfig
from tracker.tracker import ObjectTracker, TrackerConfig, TrackerType
from draw.draw import DrawConfig, FrameAnnotator
from utils.video_io import (
    VideoReader, VideoWriter,
    download_video, is_url,
    mux_audio_into_video,
)
from utils.logger_utils import FPSCounter, TrackingLogger

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

# ── constants ──────────────────────────────────────────────────────────────
MAX_FRAMES_DEMO = 900      # cap frames processed in web UI (≈30 s @ 30fps)
WORK_DIR = Path(tempfile.mkdtemp(prefix="mot_app_"))

SUPPORTED_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "sports ball", "skateboard", "surfboard", "horse", "dog",
]


# ── core processing function ───────────────────────────────────────────────

def run_tracking(
    video_file,           # Gradio File object or None
    video_url: str,
    model_name: str,
    tracker_type: str,
    conf_thresh: float,
    target_classes: list,
    frame_skip: int,
    show_trail: bool,
    trail_length: int,
    include_audio: bool,
    max_frames: int,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Main processing callback for the Gradio interface.
    Returns (output_video_path, log_text, csv_path).
    """
    progress(0, desc="Setting up …")

    # ── resolve input ──────────────────────────────────────────────────────
    if video_file is not None:
        source_path = video_file.name if hasattr(video_file, "name") else str(video_file)
    elif video_url and video_url.strip():
        try:
            progress(0.05, desc="Downloading video …")
            source_path = download_video(video_url.strip(), output_dir=str(WORK_DIR))
        except Exception as exc:
            return None, f"❌ Download failed:\n{exc}", None
    else:
        return None, "❌ Please upload a video file or enter a URL.", None

    if not Path(source_path).is_file():
        return None, f"❌ File not found: {source_path}", None

    # ── output paths ───────────────────────────────────────────────────────
    stem        = Path(source_path).stem
    silent_tmp  = str(WORK_DIR / f"{stem}_silent.mp4")
    final_out   = str(WORK_DIR / f"{stem}_tracked.mp4")
    csv_path    = str(WORK_DIR / f"{stem}_tracking.csv")

    # ── build pipeline components ──────────────────────────────────────────
    progress(0.08, desc="Loading YOLO model …")
    try:
        det_cfg = DetectorConfig(
            model_name=model_name,
            confidence_threshold=conf_thresh,
            target_classes=target_classes if target_classes else None,
            device="auto",
        )
        detector = Detector(det_cfg)
    except Exception as exc:
        return None, f"❌ Detector load failed:\n{exc}", None

    trk_cfg = TrackerConfig(tracker_type=TrackerType(tracker_type))
    tracker = ObjectTracker(trk_cfg)

    draw_cfg  = DrawConfig(show_trail=show_trail, trail_length=trail_length)
    annotator = FrameAnnotator(draw_cfg)

    # ── open video ─────────────────────────────────────────────────────────
    try:
        reader = VideoReader(source_path)
    except Exception as exc:
        return None, f"❌ Cannot open video:\n{exc}", None

    total_frames = min(reader.total_frames, max_frames)
    writer    = VideoWriter(silent_tmp, reader.fps, reader.width, reader.height)
    track_log = TrackingLogger(csv_path)
    fps_ctr   = FPSCounter(window=30)

    log_lines = [
        f"📹 Video      : {Path(source_path).name}",
        f"📐 Resolution : {reader.width}×{reader.height} @ {reader.fps:.1f} fps",
        f"🎯 Model      : {model_name}",
        f"🔁 Tracker    : {tracker_type}",
        f"🎛 Confidence : {conf_thresh}",
        f"🏷 Classes    : {target_classes or 'ALL'}",
        f"⏩ Frame skip : {frame_skip}",
        "─" * 50,
    ]

    last_tracked = []

    try:
        for frame_id, frame in reader.frames(end_frame=total_frames):
            is_det = (frame_skip == 0) or (frame_id % (frame_skip + 1) == 0)
            if is_det:
                dets         = detector.detect(frame)
                last_tracked = tracker.update(dets, frame)
            tracked = last_tracked

            cur_fps   = fps_ctr.tick()
            annotated = annotator.annotate(frame, tracked, fps=cur_fps, frame_id=frame_id)
            annotator.clear_lost_tracks({o.track_id for o in tracked})
            writer.write(annotated)

            if is_det:
                track_log.log_frame(frame_id, tracked)

            if frame_id % 50 == 0:
                pct = frame_id / max(total_frames, 1)
                progress(0.10 + pct * 0.75, desc=f"Frame {frame_id}/{total_frames} | {len(tracked)} objects")

    except Exception as exc:
        logger.exception("Frame loop error")
        log_lines.append(f"⚠️  Error at frame {frame_id}: {exc}")
    finally:
        reader.release()
        writer.release()

    # ── audio mux ─────────────────────────────────────────────────────────
    progress(0.87, desc="Muxing audio …")
    audio_ok = False
    if include_audio:
        audio_ok = mux_audio_into_video(
            silent_video=silent_tmp,
            source_with_audio=source_path,
            output_path=final_out,
        )
    if not audio_ok:
        shutil.copy2(silent_tmp, final_out)
    try:
        Path(silent_tmp).unlink(missing_ok=True)
    except Exception:
        pass

    # ── finalise log ──────────────────────────────────────────────────────
    track_log.save_json()
    track_log.close()
    summary = track_log.summary()

    log_lines += [
        f"✅ Done!",
        f"   Total detections : {summary.get('total_detections', 0)}",
        f"   Unique tracks    : {summary.get('unique_tracks', 0)}",
        f"   Frames processed : {summary.get('frames_processed', 0)}",
        f"   Audio included   : {'yes' if audio_ok else 'no (FFmpeg needed)'}",
    ]

    progress(1.0, desc="Complete!")
    return final_out, "\n".join(log_lines), csv_path


# ── Gradio UI ──────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Multi-Object Tracking",
        theme=gr.themes.Soft(primary_hue="emerald"),
        css="""
        .title-box { text-align:center; padding: 1rem 0 0.5rem; }
        .title-box h1 { font-size: 2rem; font-weight: 800; }
        .title-box p  { color: #6b7280; font-size: 0.95rem; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.HTML("""
        <div class="title-box">
            <h1>🎯 Multi-Object Detection & Tracking</h1>
            <p>YOLOv8 + ByteTrack / DeepSORT — upload a video or paste a YouTube URL</p>
        </div>
        """)

        with gr.Row():
            # ── LEFT: inputs ───────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                video_file = gr.File(
                    label="Upload video (MP4 / AVI / MOV)",
                    file_types=["video"],
                )
                video_url = gr.Textbox(
                    label="Or paste a YouTube / public video URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )

                gr.Markdown("### ⚙️ Detection")
                model_name = gr.Dropdown(
                    choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt",
                             "yolo11n.pt","yolo11s.pt"],
                    value="yolov8n.pt",
                    label="YOLO Model  (n=fastest, m=balanced)",
                )
                conf_thresh = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                        label="Confidence Threshold")
                target_classes = gr.CheckboxGroup(
                    choices=SUPPORTED_CLASSES,
                    value=["person"],
                    label="Classes to Track  (uncheck all = track everything)",
                )

                gr.Markdown("### 🔁 Tracking")
                tracker_type = gr.Radio(
                    choices=["bytetrack", "deepsort"],
                    value="bytetrack",
                    label="Tracker",
                )
                frame_skip = gr.Slider(0, 5, value=0, step=1,
                                       label="Frame Skip  (0 = every frame)")

                gr.Markdown("### 🎨 Visualization")
                show_trail   = gr.Checkbox(value=True, label="Show motion trails")
                trail_length = gr.Slider(10, 80, value=40, step=5,
                                         label="Trail length (frames)")
                include_audio = gr.Checkbox(value=True,
                                            label="Preserve original audio (requires FFmpeg)")
                max_frames = gr.Slider(100, MAX_FRAMES_DEMO, value=300, step=50,
                                       label="Max frames to process")

                run_btn = gr.Button("🚀 Run Tracking", variant="primary", size="lg")

            # ── RIGHT: outputs ─────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                out_video = gr.Video(label="Annotated Video")
                out_log   = gr.Textbox(label="Processing Log", lines=14,
                                       interactive=False)
                out_csv   = gr.File(label="Download Tracking CSV")

        gr.Markdown("""
        ---
        **Tips**
        - *YOLOv8n* is fastest; use *YOLOv8s/m* for harder scenes
        - *ByteTrack* is recommended for most videos; *DeepSORT* adds appearance re-ID
        - Frame skip = 1 processes every other frame (2× faster)
        - Audio preservation requires **FFmpeg** installed on the server
        - Tracking logs (CSV / JSON) are saved alongside the output video
        """)

        run_btn.click(
            fn=run_tracking,
            inputs=[
                video_file, video_url, model_name, tracker_type,
                conf_thresh, target_classes, frame_skip,
                show_trail, trail_length, include_audio, max_frames,
            ],
            outputs=[out_video, out_log, out_csv],
        )

    return demo


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,          # set True for a temporary public link
    )
