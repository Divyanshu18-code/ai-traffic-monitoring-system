"""
Unit tests for Video Input & Processing Pipeline (Phase 3).

Tests video file opening, metadata extraction, validation of corrupt or invalid paths,
incremental frame generation, detector integration, progress reporting,
output video generation, and resource lifecycle management.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import cv2
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

from core.detector import YOLODetector
from core.video_processor import (
    ProcessingProgress,
    ProcessingResult,
    VideoProcessor,
)
from utils.video_utils import generate_synthetic_traffic_video


@pytest.fixture(scope="module")
def shared_detector() -> YOLODetector:
    """Fixture providing initialized YOLODetector."""
    return YOLODetector(confidence=0.35)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """Generate a small 2-second synthetic video for testing."""
    temp_dir = tmp_path_factory.mktemp("video_test")
    video_path = temp_dir / "test_feed.mp4"
    return generate_synthetic_traffic_video(
        output_path=video_path,
        width=320,
        height=240,
        fps=15,
        duration_sec=2,
    )


class TestVideoProcessorValidation:
    """Test suite for video validation, opening, and metadata extraction."""

    def test_open_nonexistent_file(self) -> None:
        """Verify error is returned when video file does not exist."""
        processor = VideoProcessor("nonexistent_video_12345.mp4")
        success, msg = processor.open()
        assert success is False
        assert "not found" in msg.lower()
        assert processor.is_opened is False

    def test_open_unsupported_extension(self, tmp_path: Path) -> None:
        """Verify error is returned for unsupported extensions (.txt, .exe, etc.)."""
        fake_file = tmp_path / "script.exe"
        fake_file.write_text("dummy binary data")
        processor = VideoProcessor(fake_file)
        success, msg = processor.open()
        assert success is False
        assert "unsupported" in msg.lower() or "extension" in msg.lower()

    def test_open_corrupted_empty_video(self, tmp_path: Path) -> None:
        """Verify 0-byte video file is rejected with clear error."""
        corrupt_file = tmp_path / "empty_corrupt.mp4"
        corrupt_file.touch()
        processor = VideoProcessor(corrupt_file)
        success, msg = processor.open()
        assert success is False
        assert "empty" in msg.lower() or "corrupt" in msg.lower() or "invalid" in msg.lower()

    def test_metadata_extraction(self, sample_video: Path) -> None:
        """Verify metadata is extracted accurately from a valid video."""
        processor = VideoProcessor(sample_video)
        success, msg = processor.open()
        assert success is True
        assert processor.is_opened is True

        meta = processor.get_metadata()
        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["fps"] == 15.0
        assert meta["total_frames"] == 30  # 15 fps * 2 sec
        assert meta["duration_sec"] > 0
        assert meta["is_live"] is False
        processor.close()
        assert processor.is_opened is False


class TestVideoProcessorExecution:
    """Test suite for frame generation, detector integration, and video writing."""

    def test_frame_generator_iteration(self, sample_video: Path) -> None:
        """Verify frame_generator yields frames incrementally."""
        processor = VideoProcessor(sample_video)
        success, _ = processor.open()
        assert success is True

        frames_read = 0
        for frame_idx, frame in processor.frame_generator():
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (240, 320, 3)
            frames_read += 1

        assert frames_read == 30
        processor.close()

    def test_process_frame_integration(
        self, sample_video: Path, shared_detector: YOLODetector
    ) -> None:
        """Verify process_frame runs YOLO detection and returns annotated frame."""
        processor = VideoProcessor(sample_video)
        processor.open()

        # Read first frame
        _, frame = next(processor.frame_generator())
        annotated, detections = processor.process_frame(frame, shared_detector)

        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == frame.shape
        assert isinstance(detections, list)
        processor.close()

    def test_process_frame_with_resizing(
        self, sample_video: Path, shared_detector: YOLODetector
    ) -> None:
        """Verify process_frame handles custom target resize dimensions."""
        processor = VideoProcessor(sample_video)
        processor.open()

        _, frame = next(processor.frame_generator())
        annotated, detections = processor.process_frame(
            frame, shared_detector, resize_dim=(480, 360)
        )

        assert annotated.shape == (360, 480, 3)
        assert isinstance(detections, list)
        processor.close()

    def test_process_frame_invalid_inputs(
        self, shared_detector: YOLODetector
    ) -> None:
        """Verify process_frame handles None and empty inputs without crashing."""
        processor = VideoProcessor("dummy.mp4")
        ann, dets = processor.process_frame(None, shared_detector)  # type: ignore[arg-type]
        assert ann is None
        assert dets == []

        ann_empty, dets_empty = processor.process_frame(np.array([]), shared_detector)
        assert ann_empty.size == 0
        assert dets_empty == []

    def test_process_video_end_to_end_and_saving(
        self, sample_video: Path, shared_detector: YOLODetector, tmp_path: Path
    ) -> None:
        """Verify process_video processes entire video, emits progress, and saves output."""
        output_name = "test_annotated_run.mp4"
        processor = VideoProcessor(
            source=sample_video,
            source_type="video_file",
            output_filename=output_name,
            save_output=True,
        )

        progress_reports: List[ProcessingProgress] = []

        def on_progress(p: ProcessingProgress) -> None:
            progress_reports.append(p)

        result: ProcessingResult = processor.process_video(
            detector=shared_detector,
            progress_callback=on_progress,
        )

        assert result.success is True
        assert result.processed_frames == 30
        assert result.total_frames == 30
        assert result.avg_fps > 0.0
        assert result.elapsed_time_sec > 0.0
        assert len(progress_reports) == 30
        assert progress_reports[-1].progress_percent == 100.0

        # Verify saved output file exists and is not empty
        assert result.output_path is not None
        out_file = Path(result.output_path)
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_resource_cleanup_via_context_manager(self, sample_video: Path) -> None:
        """Verify context manager automatically releases VideoCapture on exit."""
        with VideoProcessor(sample_video) as proc:
            assert proc.is_opened is True
            assert proc.cap is not None

        assert proc.is_opened is False
        assert proc.cap is None
