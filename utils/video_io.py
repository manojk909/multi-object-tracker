"""
video_io.py - Video Input/Output Utilities

Audio fix
---------
OpenCV VideoWriter has NO audio support — it writes silent video.
We fix this by:
  1. Writing annotated frames to a temp silent MP4 (via OpenCV)
  2. After the frame loop, FFmpeg muxes the original audio track into
     the silent video → final output MP4 with audio preserved.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube|youtu|youtube-nocookie)\.(com|be)/"
    r"(watch\?v=|embed/|v/|.+\?v=)?(?P<id>[A-Za-z0-9\-=_]{11})"
)
_GENERIC_URL_RE = re.compile(r"^https?://")


def is_url(source: str) -> bool:
    return bool(_GENERIC_URL_RE.match(source))


def is_youtube_url(source: str) -> bool:
    return bool(_YOUTUBE_RE.match(source))


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Audio muxing (THE core audio fix)
# ---------------------------------------------------------------------------

def mux_audio_into_video(
    silent_video: str,
    source_with_audio: str,
    output_path: str,
    start_time: float = 0.0,
) -> bool:
    """
    Merge audio from source_with_audio into silent_video and write to output_path.

    OpenCV VideoWriter cannot encode audio. This function uses FFmpeg to
    copy the original audio track (no re-encode of video) into the
    annotated silent video, producing a complete MP4 with preserved audio.

    Parameters
    ----------
    silent_video       : annotated but silent MP4 from OpenCV
    source_with_audio  : original video file that has audio
    output_path        : destination for final video-with-audio
    start_time         : audio offset in seconds (for trimmed processing)

    Returns True on success, False if FFmpeg unavailable or failed.
    """
    if not _ffmpeg_available():
        logger.warning(
            "FFmpeg not found on PATH — audio will NOT be in the output.\n"
            "  Ubuntu/Debian : sudo apt install ffmpeg\n"
            "  macOS         : brew install ffmpeg\n"
            "  Windows       : https://www.gyan.dev/ffmpeg/builds/\n"
            "Keeping silent video at: %s", output_path
        )
        shutil.copy2(silent_video, output_path)
        return False

    tmp_output = output_path + ".tmp_mux.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video,           # annotated video (no audio)
        "-ss", str(start_time),       # seek audio to match processing start
        "-i", source_with_audio,      # original source (has audio)
        "-map", "0:v:0",              # video stream from annotated file
        "-map", "1:a:0?",             # audio stream from original (? = optional)
        "-c:v", "copy",               # copy video — no re-encode, fast
        "-c:a", "aac",                # re-encode audio as AAC for compatibility
        "-b:a", "192k",
        "-shortest",                  # trim to shortest stream
        "-movflags", "+faststart",    # web-optimised atom placement
        tmp_output,
    ]

    logger.info("Muxing original audio into annotated video …")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("FFmpeg mux failed:\n%s", result.stderr[-2000:])
            shutil.copy2(silent_video, output_path)
            return False
        shutil.move(tmp_output, output_path)
        logger.info("Audio mux complete → %s", output_path)
        return True
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out — keeping silent video")
        shutil.copy2(silent_video, output_path)
        return False
    except Exception as exc:
        logger.error("Audio mux error: %s", exc)
        shutil.copy2(silent_video, output_path)
        return False


# ---------------------------------------------------------------------------
# Video downloader
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Optional[str] = None) -> str:
    """Download a video from YouTube or public URL using yt-dlp."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mot_download_")

    output_path = os.path.join(output_dir, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        url,
    ]

    logger.info("Downloading video from: %s", url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(result.stdout)
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Run: pip install yt-dlp")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"yt-dlp failed:\n{exc.stderr}") from exc

    mp4_files = list(Path(output_dir).glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError(f"No MP4 found in {output_dir} after download.")
    downloaded = str(sorted(mp4_files)[-1])
    logger.info("Downloaded: %s", downloaded)
    return downloaded


# ---------------------------------------------------------------------------
# Video reader
# ---------------------------------------------------------------------------

class VideoReader:
    """OpenCV VideoCapture wrapper with metadata properties."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip: int = 0,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_id, frame_bgr) tuples."""
        if start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if end_frame is not None and frame_id >= end_frame:
                break
            if skip == 0 or frame_id % (skip + 1) == 0:
                yield frame_id, frame
            frame_id += 1

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoReader(path={self.path!r}, "
            f"{self.width}x{self.height} @ {self.fps:.1f}fps, "
            f"{self.total_frames} frames)"
        )


# ---------------------------------------------------------------------------
# Video writer  (writes SILENT video; audio added post-loop via mux_audio)
# ---------------------------------------------------------------------------

class VideoWriter:
    """
    Writes annotated frames to a temporary silent MP4.

    Audio is added AFTER the frame loop via mux_audio_into_video().
    The final path passed here will hold the silent version; the caller
    is responsible for muxing audio if desired.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info("VideoWriter: %s | %dx%d @ %.1ffps", output_path, width, height, fps)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer.isOpened():
            self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
