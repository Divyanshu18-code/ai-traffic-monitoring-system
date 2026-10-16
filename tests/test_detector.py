"""
Unit tests for YOLO Object Detector Module (Phase 2).

Tests the Detection dataclass, device resolution, model initialization,
detection parsing, error handling on invalid frames, and visualization overlays.
"""

from __future__ import annotations

import os
import sys
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

import torch

from core.detector import Detection, YOLODetector
from config.settings import COCO_TRAFFIC_CLASSES, get_config


class TestDetectionDataclass:
    """Test suite for Detection structured data model."""

    def test_detection_attributes_and_properties(self) -> None:
        """Verify Detection fields, width, height, and bottom_center calculations."""
        det = Detection(
            class_id=2,
            class_name="car",
            confidence=0.88,
            bbox=(100, 150, 300, 450),
            center=(200, 300),
            track_id=42,
        )

        assert det.class_id == 2
        assert det.class_name == "car"
        assert det.confidence == 0.88
        assert det.bbox == (100, 150, 300, 450)
        assert det.center == (200, 300)
        assert det.track_id == 42

        # Calculated geometry properties
        assert det.width == 200
        assert det.height == 300
        assert det.bottom_center == (200, 450)

    def test_detection_optional_track_id(self) -> None:
        """Verify Detection works without track_id assigned."""
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.72,
            bbox=(50, 60, 100, 180),
            center=(75, 120),
        )
        assert det.track_id is None
        assert det.width == 50
        assert det.height == 120
        assert det.bottom_center == (75, 180)


@pytest.fixture(scope="module")
def shared_detector() -> YOLODetector:
    """Fixture providing initialized YOLODetector instance."""
    return YOLODetector(confidence=0.35)


class TestYOLODetector:
    """Test suite for YOLODetector initialization, inference, and visualization."""

    def test_detector_initialization(self, shared_detector: YOLODetector) -> None:
        """Verify detector initializes with valid model, device, and target classes."""
        assert shared_detector.is_ready() is True
        assert shared_detector.model is not None
        assert shared_detector.device in ("cpu", "cuda")
        assert shared_detector.confidence == 0.35

        # Check standard COCO traffic class coverage
        for cls_id in (0, 2, 3, 5, 7):
            assert cls_id in shared_detector.target_classes

    def test_device_resolution_fallback(self, shared_detector: YOLODetector) -> None:
        """Verify device resolution safely handles invalid or unsupported device requests."""
        assert shared_detector._resolve_device("cpu") == "cpu"
        assert shared_detector._resolve_device("unknown_device") == "cpu"
        assert shared_detector._resolve_device("") == "cpu"
        assert shared_detector._resolve_device(None) == "cpu"

    def test_model_path_resolution_errors(self, shared_detector: YOLODetector) -> None:
        """Verify path resolution raises appropriate errors for missing or empty inputs."""
        with pytest.raises(ValueError):
            shared_detector._resolve_model_path("")

        with pytest.raises(FileNotFoundError):
            shared_detector._resolve_model_path("completely_nonexistent_custom_model_xyz.bin")

    def test_detect_invalid_frames(self, shared_detector: YOLODetector) -> None:
        """Verify detect() gracefully handles None, empty, or non-image inputs."""
        assert shared_detector.detect(None) == []
        assert shared_detector.detect(np.array([])) == []
        assert shared_detector.detect("not_an_array") == []  # type: ignore[arg-type]
        assert shared_detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    def test_detect_blank_frame(self, shared_detector: YOLODetector) -> None:
        """Verify detect() runs on a blank image without error and returns a list."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = shared_detector.detect(blank_frame)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_draw_detections_rendering(self, shared_detector: YOLODetector) -> None:
        """Verify draw_detections renders bounding boxes and labels onto the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_detections = [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.91,
                bbox=(50, 50, 200, 200),
                center=(125, 125),
                track_id=1,
            ),
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.78,
                bbox=(250, 100, 320, 280),
                center=(285, 190),
                track_id=2,
            ),
        ]

        annotated = shared_detector.draw_detections(frame, mock_detections, show_track_id=True, show_confidence=True)

        assert annotated.shape == frame.shape
        assert annotated.dtype == frame.dtype
        # Since original was all black, annotated should contain drawn pixels
        assert np.count_nonzero(annotated) > 0

    def test_draw_detections_empty_and_invalid(self, shared_detector: YOLODetector) -> None:
        """Verify draw_detections handles empty detection lists and invalid frames safely."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Empty detections returns identical copy
        annotated = shared_detector.draw_detections(frame, [])
        assert np.array_equal(annotated, frame)

        # Invalid frame returns input safely
        assert shared_detector.draw_detections(None, []) is None
        assert shared_detector.draw_detections("invalid", []) == "invalid"  # type: ignore[arg-type]

    def test_get_model_info_structure(self, shared_detector: YOLODetector) -> None:
        """Verify get_model_info returns expected configuration keys."""
        info = shared_detector.get_model_info()
        assert isinstance(info, dict)
        assert "model_path" in info
        assert "device" in info
        assert "confidence_threshold" in info
        assert "iou_threshold" in info
        assert "imgsz" in info
        assert "is_ready" in info
        assert "target_classes" in info
        assert info["is_ready"] is True
