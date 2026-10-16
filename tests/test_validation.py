"""
Unit tests for input validation, security checks, and path utilities.
"""

from __future__ import annotations

import pytest

from utils.file_utils import format_bytes, sanitize_filename
from utils.logger import sanitize_url
from utils.validation import (
    validate_confidence_threshold,
    validate_file_size,
    validate_line_ratio,
    validate_stream_url,
    validate_video_file_extension,
)


def test_validate_video_file_extension():
    """Test allowed and disallowed video file extensions."""
    valid_files = ["traffic.mp4", "cctv.avi", "camera_feed.MOV", "stream.mkv"]
    for f in valid_files:
        ok, msg = validate_video_file_extension(f)
        assert ok is True, f"Expected {f} to be valid, got: {msg}"

    invalid_files = ["script.sh", "malware.exe", "data.csv", "image.png", "no_extension"]
    for f in invalid_files:
        ok, msg = validate_video_file_extension(f)
        assert ok is False, f"Expected {f} to be invalid"
        assert len(msg) > 0


def test_validate_file_size():
    """Test file size boundaries."""
    # 0 bytes
    ok, msg = validate_file_size(0, max_mb=100)
    assert ok is False
    assert "empty" in msg.lower()

    # Normal 50MB
    ok, msg = validate_file_size(50 * 1024 * 1024, max_mb=100)
    assert ok is True

    # Exceeding 150MB with 100MB max
    ok, msg = validate_file_size(150 * 1024 * 1024, max_mb=100)
    assert ok is False
    assert "exceeds" in msg.lower()


def test_validate_stream_url():
    """Test camera stream URL validation for security and protocols."""
    valid_streams = [
        "rtsp://192.168.1.100:554/live",
        "rtsp://admin:pass@10.0.0.1/ch0",
        "rtsps://camera.lan/stream1",
        "http://traffic.example.com/live.m3u8",
        "https://secure-cam.local/feed",
    ]
    for url in valid_streams:
        ok, msg = validate_stream_url(url)
        assert ok is True, f"Expected {url} to be valid, got: {msg}"

    invalid_streams = [
        "",
        "   ",
        "file:///etc/passwd",
        "file://C:/Windows/System32",
        "javascript:alert(1)",
        "ftp://camera.example.com/stream",
        "not_a_url",
    ]
    for url in invalid_streams:
        ok, msg = validate_stream_url(url)
        assert ok is False, f"Expected {url} to be invalid"


def test_validate_line_ratio():
    """Test line ratio boundaries."""
    assert validate_line_ratio(0.50)[0] is True
    assert validate_line_ratio(0.10)[0] is True
    assert validate_line_ratio(0.90)[0] is True

    assert validate_line_ratio(0.01)[0] is False
    assert validate_line_ratio(0.99)[0] is False
    assert validate_line_ratio(-0.5)[0] is False
    assert validate_line_ratio(1.5)[0] is False


def test_validate_confidence_threshold():
    """Test confidence threshold limits."""
    assert validate_confidence_threshold(0.35)[0] is True
    assert validate_confidence_threshold(0.80)[0] is True

    assert validate_confidence_threshold(0.01)[0] is False
    assert validate_confidence_threshold(1.0)[0] is False
    assert validate_confidence_threshold(-0.1)[0] is False


def test_sanitize_filename():
    """Test directory traversal and illegal characters are stripped."""
    assert sanitize_filename("../../../etc/passwd.mp4") == "passwd.mp4"
    assert sanitize_filename("..\\..\\secret.avi") == "secret.avi"
    assert sanitize_filename("traffic test (1)!.mp4") == "traffic_test__1__.mp4"
    # Empty filename fallback
    empty_result = sanitize_filename("...")
    assert empty_result.startswith("unnamed_")


def test_sanitize_url():
    """Test password masking in RTSP/HTTP URLs."""
    raw_url = "rtsp://admin:SuperSecret123@192.168.1.55:554/h264"
    sanitized = sanitize_url(raw_url)
    assert "SuperSecret123" not in sanitized
    assert "admin:***@" in sanitized
    assert "192.168.1.55:554/h264" in sanitized


def test_format_bytes():
    """Test byte formatting."""
    assert format_bytes(500) == "500.0 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"
