"""
Input validation utilities for AI Traffic Monitoring System.

Validates uploaded files, RTSP/HTTP streams, line ratios, and configuration
parameters to ensure system security and operational integrity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2

from config.settings import get_config


def validate_video_file_extension(filename: str) -> Tuple[bool, str]:
    """
    Validate that the uploaded video file has an allowed extension.

    Args:
        filename: Name of the file.

    Returns:
        (is_valid, error_message)
    """
    config = get_config()
    ext = Path(filename).suffix.lower()
    if not ext:
        return False, "File has no extension. Please upload a valid video file."
    if ext not in config.allowed_extensions:
        allowed = ", ".join(config.allowed_extensions)
        return False, f"Unsupported file type '{ext}'. Allowed formats: {allowed}"
    return True, ""


def validate_file_size(size_bytes: int, max_mb: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate that the file size is within acceptable bounds.

    Args:
        size_bytes: Size in bytes.
        max_mb: Maximum size in MB (defaults to config limit).

    Returns:
        (is_valid, error_message)
    """
    config = get_config()
    limit_mb = max_mb if max_mb is not None else config.max_upload_size_mb
    max_bytes = limit_mb * 1024 * 1024

    if size_bytes <= 0:
        return False, "Uploaded file is empty (0 bytes)."
    if size_bytes > max_bytes:
        actual_mb = size_bytes / (1024 * 1024)
        return False, f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit ({limit_mb} MB)."
    return True, ""


def validate_video_integrity(video_path: Union[str, Path]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Attempt to open video file with OpenCV and inspect properties.

    Args:
        video_path: Path to video file.

    Returns:
        (is_valid, message, metadata_dict)
    """
    path_str = str(video_path)
    if not Path(path_str).exists():
        return False, f"Video file does not exist: {path_str}", {}

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        cap.release()
        return False, "Failed to open video file. File may be corrupted or codec unsupported.", {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first test frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, "Could not decode any frame from the video file.", {}

    if width <= 0 or height <= 0:
        return False, f"Invalid video dimensions: {width}x{height}", {}

    # If fps is missing or reported as 0, fallback to default 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0

    metadata = {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if total_frames > 0 else 0.0,
    }

    return True, "Video verified successfully.", metadata


def validate_stream_url(url: str) -> Tuple[bool, str]:
    """
    Validate camera stream URL for supported protocols and security.

    Args:
        url: RTSP or HTTP/HTTPS stream URL.

    Returns:
        (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "Stream URL cannot be empty."

    url = url.strip()
    # Check for disallowed schemes or local file traversal
    disallowed = ["file:", "javascript:", "data:", "ftp:"]
    for prefix in disallowed:
        if url.lower().startswith(prefix):
            return False, f"Disallowed scheme in camera URL: {prefix}"

    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("rtsp", "rtsps", "http", "https"):
            return False, f"Unsupported stream protocol '{parsed.scheme}'. Supported: rtsp, rtsps, http, https"
        if not parsed.netloc:
            return False, "Invalid stream URL: missing host/network location."
    except Exception as e:
        return False, f"Malformed stream URL: {e}"

    return True, ""


def validate_line_ratio(ratio: float) -> Tuple[bool, str]:
    """Validate counting line position ratio (between 0.05 and 0.95)."""
    if not (0.05 <= ratio <= 0.95):
        return False, f"Line ratio must be between 0.05 and 0.95 (received {ratio:.2f})."
    return True, ""


def validate_confidence_threshold(conf: float) -> Tuple[bool, str]:
    """Validate model confidence threshold (between 0.05 and 0.99)."""
    if not (0.05 <= conf <= 0.99):
        return False, f"Confidence threshold must be between 0.05 and 0.99 (received {conf:.2f})."
    return True, ""
