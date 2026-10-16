"""
Video Input and Processing Module for AI Traffic Monitoring System.

Handles video ingestion from uploaded files, local webcams, and authorized IP/CCTV
streams with incremental frame extraction, metadata probing, safe output encoding,
and resource lifecycle management.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import cv2
import numpy as np

from config.settings import PROCESSED_DIR, get_config
from utils.file_utils import generate_unique_filepath
from utils.logger import sanitize_url, setup_logger
from utils.validation import (
    validate_file_size,
    validate_stream_url,
    validate_video_file_extension,
    validate_video_integrity,
)
from utils.video_utils import get_best_video_writer

logger = setup_logger("video_processor")


class VideoProcessor:
    """
    Manages video ingestion, incremental frame reading, and video writing
    for files, webcams, and network streams.
    """

    def __init__(
        self,
        source: Union[str, int, Path],
        source_type: str = "video_file",
        output_filename: Optional[str] = None,
        save_output: bool = False,
    ) -> None:
        """
        Initialize VideoProcessor.

        Args:
            source: Path to video file, webcam integer index (e.g. 0), or RTSP/HTTP URL.
            source_type: 'video_file', 'webcam', or 'ip_camera'.
            output_filename: Optional name for saving processed video.
            save_output: Whether to encode and save annotated output video.
        """
        self.config = get_config()
        self.source = source
        self.source_type = source_type
        self.save_output = save_output
        self.output_filename = output_filename

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None

        self.width: int = 0
        self.height: int = 0
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.duration_sec: float = 0.0
        self.is_opened: bool = False

    def open(self) -> Tuple[bool, str]:
        """
        Open the video source and extract stream metadata.

        Returns:
            (success, status_or_error_message)
        """
        self.close()  # Ensure any previous session is released

        if self.source_type == "video_file":
            path = Path(str(self.source))
            if not path.exists():
                return False, f"Video file not found: {path}"

            ext_ok, ext_msg = validate_video_file_extension(path.name)
            if not ext_ok:
                return False, ext_msg

            size_ok, size_msg = validate_file_size(path.stat().st_size)
            if not size_ok:
                return False, size_msg

            integrity_ok, integrity_msg, meta = validate_video_integrity(path)
            if not integrity_ok:
                return False, integrity_msg

            self.cap = cv2.VideoCapture(str(path))
            self.width = meta.get("width", 640)
            self.height = meta.get("height", 480)
            self.fps = meta.get("fps", 30.0)
            self.total_frames = meta.get("total_frames", 0)
            self.duration_sec = meta.get("duration_sec", 0.0)

        elif self.source_type == "webcam":
            try:
                cam_index = int(self.source)
            except (ValueError, TypeError):
                cam_index = 0

            logger.info("Opening local webcam device: %d", cam_index)
            # Use DirectShow backend on Windows for faster initialization
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(cam_index)

            if not self.cap.isOpened():
                return False, f"Could not connect to webcam device index {cam_index}. Check permissions or device connection."

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
            if self.fps <= 0 or self.fps > 120:
                self.fps = 30.0
            self.total_frames = -1  # Live feed has indefinite frames
            self.duration_sec = 0.0

        elif self.source_type == "ip_camera":
            stream_url = str(self.source).strip()
            url_ok, url_msg = validate_stream_url(stream_url)
            if not url_ok:
                return False, url_msg

            sanitized = sanitize_url(stream_url)
            logger.info("Connecting to IP camera stream: %s", sanitized)

            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened():
                return False, f"Failed to connect to IP camera stream at {sanitized}."

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
            if self.fps <= 0 or self.fps > 120:
                self.fps = 25.0
            self.total_frames = -1
            self.duration_sec = 0.0
        else:
            return False, f"Unsupported source type: {self.source_type}"

        if not self.cap.isOpened():
            return False, "Failed to initialize OpenCV VideoCapture."

        self.is_opened = True

        # Initialize output writer if saving requested
        if self.save_output:
            out_name = self.output_filename or f"output_{Path(str(self.source)).stem}.mp4"
            target_path = generate_unique_filepath(PROCESSED_DIR, out_name, prefix="proc")
            self.writer, self.output_path = get_best_video_writer(
                target_path, self.fps, self.width, self.height
            )
            if self.writer is None:
                logger.warning("Could not initialize video writer. Output will not be saved.")
            else:
                logger.info("Initialized output video writer at: %s", self.output_path)

        return True, "Video source opened successfully."

    def get_metadata(self) -> Dict[str, Any]:
        """Return video properties and stream metadata."""
        return {
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "is_live": self.total_frames <= 0,
            "output_path": str(self.output_path) if self.output_path else None,
        }

    def frame_generator(
        self,
        frame_skip: int = 0,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield frames incrementally from the video source.

        Args:
            frame_skip: Number of frames to skip between yields (0 = process every frame).
            stop_check: Optional callable that returns True when processing should halt.

        Yields:
            (frame_index, bgr_frame_numpy_array)
        """
        if not self.is_opened or self.cap is None:
            logger.error("Cannot read frames: video source is not open.")
            return

        frame_idx = 0
        consecutive_failures = 0
        max_failures = 10 if self.source_type != "video_file" else 3

        while True:
            if stop_check and stop_check():
                logger.info("Frame generation stopped by user request.")
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.info("End of stream reached or maximum read failures reached.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Frame skip logic
            if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
                frame_idx += 1
                continue

            yield frame_idx, frame
            frame_idx += 1

    def write_processed_frame(self, frame: np.ndarray) -> bool:
        """
        Write an annotated frame to the output video file.

        Args:
            frame: Annotated BGR frame.

        Returns:
            True if written, False otherwise.
        """
        if self.writer is not None and self.writer.isOpened():
            try:
                # Resize if frame dimensions do not match writer format
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self.writer.write(frame)
                return True
            except Exception as e:
                logger.error("Error writing frame to output video: %s", e)
                return False
        return False

    def close(self) -> None:
        """Cleanly release OpenCV capture and writer resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture: %s", e)
            self.cap = None

        if self.writer is not None:
            try:
                self.writer.release()
            except Exception as e:
                logger.warning("Error releasing VideoWriter: %s", e)
            self.writer = None

        self.is_opened = False
        logger.debug("VideoProcessor resources released.")

    def __enter__(self) -> VideoProcessor:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
