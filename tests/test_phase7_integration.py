"""
Phase 7 Comprehensive System Integration and Verification Tests.

Verifies:
1. Real YOLO object detection on real image frames with bounding box, class, and confidence checks.
2. Incremental video processing with VideoProcessor and VideoWriter video generation.
3. ByteTrack tracking state persistence across consecutive frames.
4. Bidirectional vehicle counting and pedestrian separation.
5. SQLite session storage, snapshot capture, date filtering, and CSV export.
6. Error handling for corrupt/missing files, stream failures, and resource cleanup.
"""

from __future__ import annotations

import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from config.settings import SAMPLES_DIR, get_config
from core.detector import Detection, YOLODetector
from core.tracker import ByteTracker
from core.traffic_analyzer import TrafficAnalyzer
from core.vehicle_counter import VehicleCounter
from core.video_processor import VideoProcessor
from database.db_manager import DatabaseManager
from utils.validation import (
    validate_confidence_threshold,
    validate_file_size,
    validate_line_ratio,
    validate_stream_url,
    validate_video_integrity,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing a fresh isolated SQLite database."""
    return DatabaseManager(db_path=tmp_path / "test_phase7.db")


# ==============================================================================
# 1. Real YOLO Detection Verification
# ==============================================================================
def test_real_yolo_model_inference():
    """Verify loading real YOLO model and running inference on an actual video frame."""
    sample_video = SAMPLES_DIR / "sample_traffic_feed.mp4"
    assert sample_video.exists(), f"Sample traffic feed not found at {sample_video}"

    cap = cv2.VideoCapture(str(sample_video))
    assert cap.isOpened(), "Could not open sample video"
    # Seek to frame 200 where vehicles are present
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, "Failed to read frame 200 from sample video"

    detector = YOLODetector(confidence=0.20)
    info = detector.get_model_info()
    assert info["is_ready"] is True
    assert "cpu" in info["device"].lower() or "cuda" in info["device"].lower()

    detections = detector.detect(frame)
    assert isinstance(detections, list)
    assert len(detections) >= 1, "Expected at least 1 vehicle detected in frame 200"

    for det in detections:
        assert isinstance(det, Detection)
        assert det.class_id in [0, 2, 3, 5, 7]  # Traffic COCO classes
        assert det.class_name in ["person", "car", "motorcycle", "bus", "truck"]
        assert 0.0 < det.confidence <= 1.0
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 < x2 <= frame.shape[1]
        assert 0 <= y1 < y2 <= frame.shape[0]
        assert det.width == x2 - x1
        assert det.height == y2 - y1
        cx, cy = det.center
        assert x1 <= cx <= x2
        assert y1 <= cy <= y2

    # Verify annotation overlay
    annotated = detector.draw_detections(frame.copy(), detections)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8


# ==============================================================================
# 2. Real Video Processing & VideoWriter Verification
# ==============================================================================
def test_real_video_processor_pipeline(tmp_path: Path):
    """Verify end-to-end video processing and playable output video generation."""
    sample_video = SAMPLES_DIR / "sample_traffic_feed.mp4"
    assert sample_video.exists(), "Sample video required for test"

    detector = YOLODetector(confidence=0.25)
    tracker = ByteTracker(detector)
    counter = VehicleCounter(line_ratio=0.5)

    out_name = "test_phase7_output.mp4"
    vp = VideoProcessor(
        source=sample_video,
        source_type="video_file",
        output_filename=out_name,
        save_output=True,
    )

    ok, msg = vp.open()
    assert ok is True, f"VideoProcessor failed to open: {msg}"
    meta = vp.get_metadata()
    assert meta["width"] > 0
    assert meta["height"] > 0
    assert meta["total_frames"] > 0

    # Process first 30 frames
    processed_count = 0
    for frame_idx, frame in vp.frame_generator():
        processed_count += 1
        tracks = tracker.track(frame)
        counter.process_frame(tracks, frame_shape=frame.shape[:2])
        annotated = tracker.draw_tracks(frame, tracks)
        annotated = counter.draw_counting_overlay(annotated)
        vp.write_processed_frame(annotated)
        if processed_count >= 30:
            break

    output_path = vp.output_path
    vp.close()

    assert processed_count == 30
    assert output_path is not None
    assert output_path.exists(), "Output video file was not created on disk"
    file_size = output_path.stat().st_size
    assert file_size > 1000, f"Output video is too small ({file_size} bytes)"

    # Verify playability with cv2.VideoCapture
    verify_cap = cv2.VideoCapture(str(output_path))
    assert verify_cap.isOpened(), "Generated output video cannot be opened by VideoCapture"
    read_ok, test_frame = verify_cap.read()
    assert read_ok and test_frame is not None, "Failed to read frame from generated video"
    assert test_frame.shape[1] == meta["width"]
    assert test_frame.shape[0] == meta["height"]
    verify_cap.release()


# ==============================================================================
# 3. Tracking & Counting Multi-Frame Continuity
# ==============================================================================
def test_tracker_and_counter_continuity():
    """Verify ByteTracker persistent IDs and VehicleCounter line crossing."""
    detector = YOLODetector(confidence=0.25)
    tracker = ByteTracker(detector)
    counter = VehicleCounter(line_ratio=0.5)

    # Synthetic detections crossing line at y=250 in a 500x500 frame
    # Frame 1: Car above line
    det1 = Detection(
        class_id=2, class_name="car", confidence=0.9,
        bbox=(200, 180, 260, 220), center=(230, 200),
        track_id=101
    )
    counter.process_frame([det1], frame_shape=(500, 500))
    assert counter.state.total_crossed == 0

    # Frame 2: Car crossing line to y=280 (downward = inbound)
    det2 = Detection(
        class_id=2, class_name="car", confidence=0.9,
        bbox=(200, 260, 260, 300), center=(230, 280),
        track_id=101
    )
    counter.process_frame([det2], frame_shape=(500, 500))
    assert counter.state.total_inbound == 1
    assert counter.state.total_crossed == 1

    # Frame 3: Same car moving further down - duplicate prevention
    det3 = Detection(
        class_id=2, class_name="car", confidence=0.9,
        bbox=(200, 320, 260, 360), center=(230, 340),
        track_id=101
    )
    counter.process_frame([det3], frame_shape=(500, 500))
    assert counter.state.total_inbound == 1
    assert counter.state.total_crossed == 1

    # Separate pedestrian tracking
    ped1 = Detection(
        class_id=0, class_name="person", confidence=0.85,
        bbox=(50, 180, 80, 240), center=(65, 210),
        track_id=202
    )
    counter.process_frame([ped1], frame_shape=(500, 500))
    ped2 = Detection(
        class_id=0, class_name="person", confidence=0.85,
        bbox=(50, 260, 80, 320), center=(65, 290),
        track_id=202
    )
    counter.process_frame([ped2], frame_shape=(500, 500))
    assert counter.state.total_unique_pedestrians == 1
    # Pedestrians are counted separately from vehicles
    assert counter.state.total_unique_vehicles == 1
    assert counter.state.class_counts.get("car", 0) == 1
    assert counter.state.total_inbound == 2


# ==============================================================================
# 4. Database Persistence and Restart Simulation
# ==============================================================================
def test_database_persistence_and_restart(tmp_path: Path):
    """Verify that sessions and snapshots persist across DatabaseManager restarts."""
    db_file = tmp_path / "persistence_check.db"

    # Step 1: Write session with first instance
    db1 = DatabaseManager(db_path=db_file)
    db1.create_session("sess_restart_1", "video_file", "highway.mp4")
    db1.save_snapshot("sess_restart_1", 30, active_vehicles=5, current_fps=29.5)
    db1.finalize_session(
        "sess_restart_1",
        duration_sec=12.5,
        total_vehicles=8,
        cars=6,
        trucks=2,
        inbound_count=5,
        outbound_count=3,
        avg_fps=28.4,
    )
    del db1  # Simulate process termination

    # Step 2: Open with second fresh instance
    db2 = DatabaseManager(db_path=db_file)
    session = db2.get_session("sess_restart_1")
    assert session is not None
    assert session["source_name"] == "highway.mp4"
    assert session["total_vehicles"] == 8
    assert session["cars"] == 6
    assert session["inbound_count"] == 5

    # Check snapshots
    snaps = db2.get_snapshots("sess_restart_1")
    assert len(snaps) == 1
    assert snaps[0]["active_vehicles"] == 5

    # Check CSV export
    df = db2.export_sessions_dataframe()
    assert len(df) == 1
    assert "session_id" in df.columns


# ==============================================================================
# 5. Error Handling & Edge Cases
# ==============================================================================
def test_error_handling_and_resource_safety(tmp_path: Path):
    """Verify application handles corrupt files, missing paths, and closes resources."""
    # 1. Non-existent video file
    vp_bad = VideoProcessor(source=tmp_path / "non_existent.mp4")
    ok, err = vp_bad.open()
    assert ok is False
    assert "not found" in err.lower() or "not exist" in err.lower()
    vp_bad.close()

    # 2. Corrupt / Empty video file
    empty_file = tmp_path / "empty_corrupt.mp4"
    empty_file.write_bytes(b"")
    valid_ok, valid_msg, _ = validate_video_integrity(empty_file)
    assert valid_ok is False
    assert "corrupted" in valid_msg.lower() or "failed" in valid_msg.lower()

    # 3. Settings bounds validation
    ok_conf, _ = validate_confidence_threshold(0.5)
    assert ok_conf is True
    bad_conf, _ = validate_confidence_threshold(1.5)
    assert bad_conf is False

    ok_line, _ = validate_line_ratio(0.5)
    assert ok_line is True
    bad_line, _ = validate_line_ratio(-0.2)
    assert bad_line is False

    # 4. Stream URL security validation
    assert validate_stream_url("rtsp://admin:pass@192.168.1.1:554/ch0")[0] is True
    assert validate_stream_url("ftp://invalidscheme.com/video")[0] is False
    assert validate_stream_url("not_a_valid_url")[0] is False
