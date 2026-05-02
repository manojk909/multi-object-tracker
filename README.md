---
title: Multi-Object Detection and Tracking
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: YOLOv8-based real-time multi-object tracking
---

## 📦 Deliverables

All required deliverables for this project have been compiled and organized in a single Google Drive folder.

🔗 **Access Link:** [View Deliverables Folder](https://drive.google.com/drive/folders/18KJAKRtE4ee6vkw-0r31TAGtJA73-e_6?usp=sharing)

The folder includes:
- Annotated output video
- Original public video
- Short technical report
- Sample screenshots of results
- Short demo video (3–5 minutes) explaining the approach


# 🎯 Multi-Object Detection & Tracking Pipeline

A production-ready, end-to-end Python pipeline for **detecting**, **tracking**, and **annotating** multiple objects in video — with **audio preserved** in the output.

🔗 **Live Demo**: [Multi Object Tracker](https://huggingface.co/spaces/mk909/multi-object-tracker)

---

## 💡 Motivation

This project demonstrates real-time multi-object tracking using modern detection + tracking pipelines, solving challenges like occlusion, ID switching, and audio preservation — common in surveillance, sports analytics, and autonomous systems.

---

## 🎥 Demo

<img width="1900" height="852" alt="Screenshot 2026-05-02 105422" src="https://github.com/user-attachments/assets/d536b3b7-ac40-44ca-9a64-c9d9d14ab665" />

<img width="1919" height="860" alt="Screenshot 2026-05-02 110217" src="https://github.com/user-attachments/assets/b55acc7f-ff31-4d75-811e-d3604801aa34" />

<img width="1885" height="852" alt="Screenshot 2026-05-02 110317" src="https://github.com/user-attachments/assets/47211be1-d58a-4b7c-bdda-f854dc49152c" />

---

## 📸 Features

- **YOLOv8 (Ultralytics)** object detection (nano → extra-large variants)
- **ByteTrack** (default) and **DeepSORT** multi-object tracking
- **Stable cross-frame IDs** with occlusion handling
- **Motion trail visualization** — fading path per object
- **Audio preserved** in output MP4 via FFmpeg muxing
- **YouTube / URL download** via yt-dlp
- **Gradio web UI** — no code needed
- **CLI interface** for batch / server use
- **CSV + JSON tracking logs** per session
- **GPU / CPU / MPS** auto-detection

---

## 🗂️ Project Structure

```
multi_object_tracker/
├── app.py                   ← Gradio web UI (Hugging Face Spaces entry point)
├── main.py                  ← CLI entry point
├── config.py                ← Central configuration dataclasses
├── packages.txt             ← System packages for HF Spaces (ffmpeg)
├── requirements.txt
│
├── detector/
│   └── detector.py          ← YOLOv8 wrapper: Detection, DetectorConfig, Detector
│
├── tracker/
│   └── tracker.py           ← ByteTrack / DeepSORT / IoU-fallback wrappers
│
├── draw/
│   └── draw.py              ← Bounding boxes, labels, trails, FPS HUD
│
├── utils/
│   ├── video_io.py          ← VideoReader, VideoWriter, mux_audio_into_video
│   └── logger_utils.py      ← FPSCounter, TrackingLogger (CSV + JSON)
│
├── report/
│   └── technical_report.md  ← 1–2 page technical write-up
│
└── output/                  ← Auto-created: annotated videos + logs
```

---

## ⚙️ Setup

### Prerequisites
- Python **3.10+**
- FFmpeg (for audio): `sudo apt install ffmpeg` · `brew install ffmpeg`

### Install

```bash
git clone https://github.com/manojk909/multi-object-tracker.git
cd multi-object-tracker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For **CUDA GPU**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1 — Web UI (Gradio)

```bash
python app.py
# → opens at http://localhost:7860
```

Upload a video or paste a YouTube URL and click **Run Tracking**.

### Option 2 — CLI

```bash
# Local file — track people
python main.py --video path/to/video.mp4 --classes person

# YouTube URL — full auto download + track + audio preserved
python main.py --video "https://www.youtube.com/watch?v=VIDEO_ID"

# GPU + larger model + live preview
python main.py --video video.mp4 --model yolov8m.pt --device cuda --display

# Track vehicles only
python main.py --video traffic.mp4 --classes car bus truck motorcycle

# DeepSORT for appearance-based re-ID
python main.py --video myvideo.mp4 --tracker deepsort

# Skip every other frame (2× faster)
python main.py --video myvideo.mp4 --skip 1

# Skip audio muxing
python main.py --video myvideo.mp4 --no-audio
```

### Full CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | *(required)* | File path or public URL |
| `--output-dir` | `output/` | Where to save results |
| `--model` | `yolov8n.pt` | YOLO weights variant |
| `--conf` | `0.35` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--classes` | all | Space-separated class names |
| `--tracker` | `bytetrack` | `bytetrack` or `deepsort` |
| `--track-buffer` | `30` | Frames before lost track deletion |
| `--skip` | `0` | Frame skip (0 = every frame) |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, `auto` |
| `--no-audio` | off | Skip FFmpeg audio muxing |
| `--no-trail` | off | Disable motion trails |
| `--display` | off | Show live preview window |

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `output/<name>_tracked.mp4` | Annotated video **with audio** |
| `output/logs/<name>_tracking.csv` | Per-frame detection log |
| `output/logs/<name>_tracking.json` | Same data grouped by frame |

### CSV schema
```
frame_id, track_id, class_id, class_name, x1, y1, x2, y2, confidence
```

---

## 🔊 Audio Preservation (Fix)

OpenCV's `VideoWriter` cannot write audio streams — it always produces silent video.

**Our fix**: After the frame loop completes, we call FFmpeg to **mux** the original audio track directly into the annotated video without re-encoding the video stream:

```
[silent annotated video] ──┐
                            ├──► FFmpeg mux ──► final video with audio ✅
[original audio stream]  ──┘
```

FFmpeg must be installed on the system (`packages.txt` handles this on HF Spaces automatically).

---

## 🔍 Model Choices

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `yolov8n.pt` | ⚡ Fastest | Good | Real-time, CPU |
| `yolov8s.pt` | Fast | Better | Balanced default |
| `yolov8m.pt` | Medium | Great | Higher accuracy |
| `yolov8l.pt` | Slow | Excellent | Offline |
| `yolov8x.pt` | Slowest | Best | Max accuracy |

Models download automatically on first use.

---

## 🔁 Tracker Choices

**ByteTrack** (default)
- Uses ALL detections including low-confidence ones → better occlusion recovery
- No appearance features needed → very fast
- Best for: sports, surveillance, traffic

**DeepSORT**
- Adds appearance embeddings (MobileNet) for re-ID after long occlusions
- Slower but more robust when objects look similar and disappear
- Best for: long occlusions, crowded scenes

---

## 🧩 Assumptions & Limitations

**Assumptions**
- Input video readable by OpenCV (MP4, AVI, MOV, MKV, …)
- YOLO model trained on COCO classes (80 categories)
- Camera motion is moderate (extreme shake degrades IoU matching)

**Limitations**
- No cross-camera tracking — IDs are per video only
- Long disappearances (> `track_buffer` frames) reset the ID
- Very small objects (< 8 px after resize) are rarely detected
- Age-restricted or DRM YouTube videos cannot be downloaded

---

## 🛠️ Programmatic API

```python
from detector.detector import Detector, DetectorConfig
from tracker.tracker import ObjectTracker, TrackerConfig
from draw.draw import FrameAnnotator

detector  = Detector(DetectorConfig(model_name="yolov8s.pt", confidence_threshold=0.4))
tracker   = ObjectTracker(TrackerConfig())
annotator = FrameAnnotator()

# In your frame loop:
detections = detector.detect(frame_bgr)
tracked    = tracker.update(detections, frame_bgr)
annotated  = annotator.annotate(frame_bgr, tracked, fps=30.0, frame_id=42)
```

---

## 🚀 Deploy to Hugging Face Spaces

1. Push this repo to GitHub
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
3. Choose **Gradio** SDK, link your GitHub repo
4. HF Spaces reads `packages.txt` → installs `ffmpeg` automatically
5. Reads `requirements.txt` → installs Python deps
6. Launches `app.py` — your app is live at `https://huggingface.co/spaces/mk909/multi-object-tracker`

---

## 📄 License

MIT
=======

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
