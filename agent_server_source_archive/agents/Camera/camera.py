"""Camera capture backends for image input.

Supports ffmpeg (v4l2), static file, and dummy backends.
Images are returned as base64-encoded JPEG strings ready for the LLM API.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

FFMPEG_BIN = os.environ.get(
    "FFMPEG_BIN", "/opt/sophon/sophon-ffmpeg-latest/bin/ffmpeg"
)


class CameraBackend(ABC):
    """Abstract camera backend."""

    @abstractmethod
    def capture(self) -> Optional[str]:
        """Capture a single frame, return base64 JPEG string or None."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is ready to capture."""
        ...


class FfmpegBackend(CameraBackend):
    """Capture from a v4l2 camera device via ffmpeg."""

    def __init__(self, device: str = "/dev/video0", width: int = 640,
                 height: int = 480, ffmpeg_bin: str = FFMPEG_BIN):
        self.device = device
        self.width = width
        self.height = height
        self.ffmpeg_bin = ffmpeg_bin

    def is_available(self) -> bool:
        if not os.path.exists(self.device):
            logger.debug(f"Camera device {self.device} not found")
            return False
        if not os.path.exists(self.ffmpeg_bin):
            logger.debug(f"ffmpeg binary {self.ffmpeg_bin} not found")
            return False
        return True

    def capture(self) -> Optional[str]:
        if not self.is_available():
            return None

        cmd = [
            self.ffmpeg_bin,
            "-f", "v4l2",
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.device,
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "2",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=5,
            )
            if proc.returncode != 0 or not proc.stdout:
                logger.warning(f"ffmpeg capture failed: {proc.stderr.decode()[:200]}")
                return None
            return self._encode_jpeg(proc.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"ffmpeg error: {e}")
            return None

    @staticmethod
    def _encode_jpeg(raw: bytes) -> str:
        """Convert raw JPEG bytes to base64 data URI string."""
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


class FileBackend(CameraBackend):
    """Read a static image file (useful for testing)."""

    def __init__(self, path: str = ""):
        self.path = path

    def is_available(self) -> bool:
        return bool(self.path) and os.path.isfile(self.path)

    def capture(self) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
            # Ensure JPEG encoding
            img = Image.open(io.BytesIO(raw))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return b64
        except Exception as e:
            logger.warning(f"FileBackend error: {e}")
            return None


class DummyBackend(CameraBackend):
    """Always returns None — no camera available."""

    def is_available(self) -> bool:
        return False

    def capture(self) -> Optional[str]:
        return None


def create_camera(backend: str = "auto", **kwargs) -> CameraBackend:
    """Factory: create the appropriate camera backend.

    Args:
        backend: "auto", "ffmpeg", "file", or "dummy"
        **kwargs: backend-specific parameters
    """
    if backend == "dummy":
        return DummyBackend()

    if backend == "file":
        return FileBackend(path=kwargs.get("path", ""))

    if backend == "ffmpeg":
        return FfmpegBackend(
            device=kwargs.get("device", "/dev/video0"),
            width=kwargs.get("width", 640),
            height=kwargs.get("height", 480),
        )

    # auto: try ffmpeg first, then file, then dummy
    ffmpeg_be = FfmpegBackend()
    if ffmpeg_be.is_available():
        return ffmpeg_be

    file_be = FileBackend(path=kwargs.get("path", ""))
    if file_be.is_available():
        return file_be

    logger.info("No camera available, using dummy backend")
    return DummyBackend()
