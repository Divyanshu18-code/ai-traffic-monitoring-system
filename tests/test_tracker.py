"""
Unit tests for ByteTrack Object Tracking Module (Phase 4).

Tests tracker initialization, parameter updates, state resetting,
trajectory history management, invalid input handling, and track rendering.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

# Ensure Windows OpenMP & Torch safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

from core.detector import Detection, YOLODetector
from core.tracker import ByteTracker


@pytest.fixture(scope="module")
def shared_detector() -> YOLODetector:
    """Fixture providing initialized YOLODetector."""
    return YOLODetector(confidence=0.35)


@pytest.fixture
def tracker(shared_detector: YOLODetector) -> ByteTracker:
    """Fixture providing clean ByteTracker instance for each test."""
    t = ByteTracker(shared_detector, max_history=30)
    t.reset()
    return t


class TestByteTracker:
    """Test suite for ByteTracker functionality and resource management."""

    def test_tracker_initialization(self, tracker: ByteTracker) -> None:
        """Verify tracker initializes with expected defaults and empty state."""
        assert tracker.tracker_type == "bytetrack.yaml"
        assert tracker.max_history == 30
        assert len(tracker.track_history) == 0
        assert len(tracker.track_classes) == 0

    def test_tracker_empty_and_invalid_inputs(self, tracker: ByteTracker) -> None:
        """Verify track() handles None, empty, or non-image inputs safely."""
        assert tracker.track(None) == []  # type: ignore[arg-type]
        assert tracker.track(np.array([])) == []
        assert tracker.track("invalid_frame") == []  # type: ignore[arg-type]
        assert tracker.track(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_tracker_state_reset(self, tracker: ByteTracker) -> None:
        """Verify reset() clears all in-memory track histories and class associations."""
        tracker.track_history[1].append((100, 100))
        tracker.track_history[1].append((100, 150))
        tracker.track_classes[1] = "car"

        assert len(tracker.track_history) == 1
        assert len(tracker.track_classes) == 1

        tracker.reset()

        assert len(tracker.track_history) == 0
        assert len(tracker.track_classes) == 0

    def test_tracker_update_config(self, tracker: ByteTracker) -> None:
        """Verify dynamic configuration updates affect tracker settings."""
        tracker.update_config(max_history=50, confidence=0.45, iou=0.50)
        assert tracker.max_history == 50
        assert tracker.detector.confidence == 0.45
        assert tracker.detector.iou == 0.50

    def test_track_history_retrieval(self, tracker: ByteTracker) -> None:
        """Verify get_track_history returns trajectory coordinates."""
        tracker.track_history[42].extend([(10, 20), (15, 30), (20, 40)])
        history = tracker.get_track_history(42)
        assert history == [(10, 20), (15, 30), (20, 40)]

        # Non-existent track ID should return empty list
        assert tracker.get_track_history(999) == []

    def test_draw_tracks_rendering(self, tracker: ByteTracker) -> None:
        """Verify draw_tracks renders bounding boxes, badges, and trajectory trails."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_detections = [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.89,
                bbox=(100, 100, 200, 200),
                center=(150, 150),
                track_id=1,
            ),
            Detection(
                class_id=5,
                class_name="bus",
                confidence=0.92,
                bbox=(300, 100, 450, 300),
                center=(375, 200),
                track_id=2,
            ),
        ]
        # Add trajectory points
        tracker.track_history[1].extend([(150, 120), (150, 135), (150, 150)])
        tracker.track_history[2].extend([(375, 160), (375, 180), (375, 200)])

        annotated = tracker.draw_tracks(frame, mock_detections, show_trails=True)
        assert annotated.shape == frame.shape
        assert annotated.dtype == frame.dtype
        assert np.count_nonzero(annotated) > 0

    def test_draw_tracks_empty_and_invalid(self, tracker: ByteTracker) -> None:
        """Verify draw_tracks handles empty lists and invalid frames safely."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert np.array_equal(tracker.draw_tracks(frame, []), frame)
        assert tracker.draw_tracks(None, []) is None  # type: ignore[arg-type]
