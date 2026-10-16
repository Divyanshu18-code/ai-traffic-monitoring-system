"""
Comprehensive unit and integration tests for SQLite database manager.

Covers all Phase 5 requirements:
1. Database initialization
2. Table creation
3. Session insertion
4. Session retrieval
5. Session updates
6. Session finalization
7. Date filtering
8. Recent session retrieval
9. Snapshot insertion
10. Foreign-key behavior & cascade deletion
11. Duplicate session handling
12. Invalid data validation
13. Database path configuration
14. Existing records surviving reinitialization
15. Historical statistics and metrics aggregation
16. VideoProcessor database integration
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from core.detector import Detection
from core.vehicle_counter import VehicleCounter
from core.video_processor import VideoProcessor
from database.db_manager import DatabaseManager


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Create a temporary SQLite database for testing using pytest tmp_path."""
    test_db_path = tmp_path / "test_traffic.db"
    return DatabaseManager(db_path=test_db_path)


# 1. Database Initialization
def test_database_initialization(temp_db: DatabaseManager):
    """Test that the database file is initialized and accessible."""
    assert temp_db.db_path.exists()
    assert temp_db.db_path.is_file()
    all_sessions = temp_db.get_all_sessions()
    assert all_sessions == []


# 2. Table Creation
def test_table_creation(temp_db: DatabaseManager):
    """Test that all required schema tables and columns are created."""
    with temp_db._get_connection() as conn:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "sessions" in tables
        assert "traffic_snapshots" in tables
        assert "system_metadata" in tables

        session_cols = {
            col["name"]
            for col in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        required_cols = {
            "session_id",
            "source_type",
            "source_name",
            "start_time",
            "end_time",
            "duration_sec",
            "total_vehicles",
            "cars",
            "buses",
            "trucks",
            "motorcycles",
            "pedestrians",
            "total_crossings",
            "inbound_count",
            "outbound_count",
            "avg_fps",
            "output_video_path",
            "status",
            "error_summary",
        }
        assert required_cols.issubset(session_cols)


# 3. Session Insertion
def test_session_insertion(temp_db: DatabaseManager):
    """Test inserting a new session record."""
    session_id = "sess_insert_001"
    created = temp_db.create_session(
        session_id=session_id,
        source_type="video_file",
        source_name="highway_test.mp4",
        status="started",
        notes="Insertion test session",
    )
    assert created is True


# 4. Session Retrieval
def test_session_retrieval(temp_db: DatabaseManager):
    """Test retrieving an existing session record by ID."""
    session_id = "sess_retrieve_002"
    temp_db.create_session(
        session_id=session_id,
        source_type="webcam",
        source_name="webcam_device_0",
        status="started",
    )
    record = temp_db.get_session(session_id)
    assert record is not None
    assert record["session_id"] == session_id
    assert record["source_type"] == "webcam"
    assert record["source_name"] == "webcam_device_0"
    assert record["status"] == "started"
    assert record["total_vehicles"] == 0

    # Non-existent ID returns None
    assert temp_db.get_session("non_existent_id") is None


# 5. Session Updates
def test_session_updates(temp_db: DatabaseManager):
    """Test updating session status and intermediate statistics."""
    session_id = "sess_update_003"
    temp_db.create_session(session_id, "video_file", "sample.mp4", status="started")

    # Update status
    status_updated = temp_db.update_session_status(
        session_id, status="in_progress", error_summary=None
    )
    assert status_updated is True
    session = temp_db.get_session(session_id)
    assert session["status"] == "in_progress"

    # Update intermediate counts
    stats_updated = temp_db.update_session_statistics(
        session_id=session_id,
        total_vehicles=8,
        cars=6,
        trucks=2,
        total_crossings=4,
        inbound_count=3,
        outbound_count=1,
        avg_fps=26.4,
    )
    assert stats_updated is True
    updated_session = temp_db.get_session(session_id)
    assert updated_session["total_vehicles"] == 8
    assert updated_session["cars"] == 6
    assert updated_session["trucks"] == 2
    assert updated_session["total_crossings"] == 4
    assert updated_session["avg_fps"] == 26.4


# 6. Session Finalization
def test_session_finalization(temp_db: DatabaseManager):
    """Test finalizing a session with complete metrics and status."""
    session_id = "sess_finalize_004"
    temp_db.create_session(session_id, "ip_camera", "rtsp://camera1", status="started")

    finalized = temp_db.finalize_session(
        session_id=session_id,
        duration_sec=45.2,
        total_vehicles=35,
        cars=25,
        buses=3,
        trucks=5,
        motorcycles=2,
        pedestrians=7,
        total_crossings=20,
        inbound_count=12,
        outbound_count=8,
        avg_fps=29.8,
        output_video_path="data/processed/output.mp4",
        status="completed",
    )
    assert finalized is True

    session = temp_db.get_session(session_id)
    assert session["status"] == "completed"
    assert session["duration_sec"] == 45.2
    assert session["total_vehicles"] == 35
    assert session["cars"] == 25
    assert session["buses"] == 3
    assert session["trucks"] == 5
    assert session["motorcycles"] == 2
    assert session["pedestrians"] == 7
    assert session["total_crossings"] == 20
    assert session["inbound_count"] == 12
    assert session["outbound_count"] == 8
    assert session["avg_fps"] == 29.8
    assert session["output_video_path"] == "data/processed/output.mp4"

    # Non-existent session returns False
    assert temp_db.finalize_session("unknown_sess") is False


# 7. Date Filtering
def test_date_filtering(temp_db: DatabaseManager):
    """Test filtering recorded sessions by date range and source type."""
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)

    temp_db.create_session("sess_today_1", "video_file", "f1.mp4", start_time=now)
    temp_db.create_session("sess_today_2", "webcam", "cam0", start_time=now)
    temp_db.create_session("sess_yest", "video_file", "f2.mp4", start_time=yesterday)
    temp_db.create_session("sess_old", "video_file", "f3.mp4", start_time=two_days_ago)

    today_str = now.strftime("%Y-%m-%d")
    today_sessions = temp_db.filter_sessions(start_date=today_str, end_date=today_str)
    assert len(today_sessions) == 2

    webcam_sessions = temp_db.filter_sessions(source_type="webcam")
    assert len(webcam_sessions) == 1
    assert webcam_sessions[0]["session_id"] == "sess_today_2"

    range_sessions = temp_db.filter_sessions(
        start_date=yesterday.strftime("%Y-%m-%d"),
        end_date=today_str,
    )
    assert len(range_sessions) == 3


# 8. Recent Session Retrieval
def test_recent_session_retrieval(temp_db: DatabaseManager):
    """Test retrieving a limited set of recent sessions ordered by start_time."""
    base_time = datetime(2026, 10, 16, 12, 0, 0)
    for i in range(5):
        temp_db.create_session(
            session_id=f"sess_recent_{i}",
            source_type="video_file",
            source_name=f"test_{i}.mp4",
            start_time=base_time + timedelta(minutes=i * 10),
        )

    recent_3 = temp_db.get_recent_sessions(limit=3)
    assert len(recent_3) == 3
    # Ordered descending: most recent should be sess_recent_4
    assert recent_3[0]["session_id"] == "sess_recent_4"
    assert recent_3[1]["session_id"] == "sess_recent_3"
    assert recent_3[2]["session_id"] == "sess_recent_2"


# 9. Snapshot Insertion
def test_snapshot_insertion(temp_db: DatabaseManager):
    """Test periodic traffic snapshot insertion and retrieval."""
    session_id = "sess_snap_test"
    temp_db.create_session(session_id, "video_file", "traffic.mp4")

    snap1 = temp_db.insert_traffic_snapshot(
        session_id=session_id,
        frame_number=30,
        active_vehicles=4,
        congestion_level="LOW",
        cumulative_inbound=2,
        cumulative_outbound=1,
        total_crossings=3,
        current_fps=28.5,
        cars=3,
        trucks=1,
    )
    assert snap1 is True

    snap2 = temp_db.save_snapshot(
        session_id=session_id,
        frame_number=60,
        active_vehicles=9,
        congestion_level="MODERATE",
        cumulative_inbound=5,
        cumulative_outbound=3,
        total_crossings=8,
        current_fps=29.2,
        cars=6,
        buses=1,
        trucks=2,
    )
    assert snap2 is True

    snapshots = temp_db.get_snapshots_for_session(session_id)
    assert len(snapshots) == 2
    assert snapshots[0]["frame_number"] == 30
    assert snapshots[0]["cars"] == 3
    assert snapshots[1]["frame_number"] == 60
    assert snapshots[1]["congestion_level"] == "MODERATE"
    assert snapshots[1]["total_crossings"] == 8


# 10. Foreign-Key Behavior & Cascade Deletion
def test_foreign_key_behavior(temp_db: DatabaseManager):
    """Test foreign key constraint enforcement and cascade deletion on sessions."""
    # Inserting snapshot for non-existent session should fail gracefully
    invalid_snap = temp_db.insert_traffic_snapshot(
        session_id="non_existent_foreign_key_session",
        frame_number=1,
        active_vehicles=2,
    )
    assert invalid_snap is False

    # Create session, add snapshot, and delete session to verify cascade delete
    session_id = "sess_to_cascade_delete"
    temp_db.create_session(session_id, "video_file", "test.mp4")
    temp_db.insert_traffic_snapshot(session_id, frame_number=10, active_vehicles=3)

    assert len(temp_db.get_snapshots_for_session(session_id)) == 1

    # Delete session
    deleted = temp_db.delete_session(session_id)
    assert deleted is True
    assert temp_db.get_session(session_id) is None
    # Snapshots should be cascade deleted
    assert len(temp_db.get_snapshots_for_session(session_id)) == 0


# 11. Duplicate Session Handling
def test_duplicate_session_handling(temp_db: DatabaseManager):
    """Test that attempting to insert a duplicate session ID returns False without error."""
    session_id = "sess_duplicate_001"
    first = temp_db.create_session(session_id, "video_file", "file.mp4")
    assert first is True

    second = temp_db.create_session(session_id, "video_file", "file.mp4")
    assert second is False


# 12. Invalid Data Validation
def test_invalid_data_validation(temp_db: DatabaseManager):
    """Test that invalid parameters raise clear ValueError exceptions."""
    # Invalid session_id
    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        temp_db.create_session("", "video_file", "test.mp4")

    # Invalid source_type
    with pytest.raises(ValueError, match="Invalid source_type"):
        temp_db.create_session("sess_inv_1", "airplane", "test.mp4")

    # Invalid status
    with pytest.raises(ValueError, match="Invalid status"):
        temp_db.create_session("sess_inv_2", "video_file", "test.mp4", status="unknown_status")

    with pytest.raises(ValueError, match="Invalid status"):
        temp_db.update_session_status("sess_inv_3", status="flying")

    # Negative counts in finalization
    temp_db.create_session("sess_neg_1", "video_file", "test.mp4")
    with pytest.raises(ValueError, match="must be a non-negative"):
        temp_db.finalize_session("sess_neg_1", total_vehicles=-5)

    with pytest.raises(ValueError, match="must be a non-negative"):
        temp_db.finalize_session("sess_neg_1", duration_sec=-1.0)


# 13. Database Path Configuration
def test_database_path_configuration(tmp_path: Path):
    """Test initializing database at a custom configurable nested path."""
    nested_dir = tmp_path / "custom_env" / "nested_subfolder"
    custom_db_path = nested_dir / "custom_traffic.db"

    # Directory does not exist yet
    assert not nested_dir.exists()

    db = DatabaseManager(db_path=custom_db_path)
    assert custom_db_path.exists()
    assert custom_db_path.is_file()
    assert db.db_path == custom_db_path


# 14. Existing Records Surviving Reinitialization
def test_existing_records_surviving_reinitialization(tmp_path: Path):
    """Test that existing records are preserved when DatabaseManager is reinitialized."""
    db_file = tmp_path / "persistent_traffic.db"

    # First session instance
    db1 = DatabaseManager(db_path=db_file)
    db1.create_session("persisted_sess_001", "video_file", "highway.mp4")
    db1.finalize_session(
        session_id="persisted_sess_001",
        total_vehicles=15,
        cars=10,
        inbound_count=8,
        outbound_count=7,
        avg_fps=25.0,
    )
    db1.insert_traffic_snapshot("persisted_sess_001", frame_number=10, active_vehicles=5)

    # Re-initialize with new manager instance pointing to same file
    db2 = DatabaseManager(db_path=db_file)
    retrieved = db2.get_session("persisted_sess_001")
    assert retrieved is not None
    assert retrieved["session_id"] == "persisted_sess_001"
    assert retrieved["total_vehicles"] == 15
    assert retrieved["cars"] == 10

    snaps = db2.get_snapshots_for_session("persisted_sess_001")
    assert len(snaps) == 1
    assert snaps[0]["frame_number"] == 10


# 15. Historical Statistics and Aggregates
def test_historical_statistics_and_aggregates(temp_db: DatabaseManager):
    """Test calculating aggregate traffic statistics across sessions."""
    temp_db.create_session("sess_agg_1", "video_file", "vid1.mp4")
    temp_db.finalize_session(
        "sess_agg_1",
        duration_sec=10.0,
        total_vehicles=12,
        cars=8,
        buses=2,
        trucks=2,
        motorcycles=0,
        pedestrians=3,
        total_crossings=10,
        inbound_count=6,
        outbound_count=4,
        avg_fps=30.0,
    )

    temp_db.create_session("sess_agg_2", "video_file", "vid2.mp4")
    temp_db.finalize_session(
        "sess_agg_2",
        duration_sec=20.0,
        total_vehicles=18,
        cars=12,
        buses=1,
        trucks=3,
        motorcycles=2,
        pedestrians=1,
        total_crossings=15,
        inbound_count=8,
        outbound_count=7,
        avg_fps=28.0,
    )

    stats = temp_db.get_historical_statistics()
    assert stats["total_sessions"] == 2
    assert stats["total_vehicles"] == 30
    assert stats["total_cars"] == 20
    assert stats["total_buses"] == 3
    assert stats["total_trucks"] == 5
    assert stats["total_motorcycles"] == 2
    assert stats["total_pedestrians"] == 4
    assert stats["total_crossings"] == 25
    assert stats["total_inbound"] == 14
    assert stats["total_outbound"] == 11
    assert stats["total_monitoring_seconds"] == 30.0


# 16. VideoProcessor Integration with Database
def test_video_processor_database_integration(tmp_path: Path, temp_db: DatabaseManager):
    """Test VideoProcessor saving session and snapshots to SQLite during processing."""
    # Create a small synthetic video
    video_path = tmp_path / "proc_test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for _ in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        Detection(
            class_id=2,
            class_name="car",
            confidence=0.85,
            bbox=(50, 50, 100, 100),
            center=(75, 75),
            track_id=1,
        )
    ]
    mock_detector.draw_detections.side_effect = lambda frame, *args, **kwargs: frame

    counter = VehicleCounter(line_ratio=0.5)
    vp = VideoProcessor(source=video_path, source_type="video_file")

    result = vp.process_video(
        detector=mock_detector,
        counter=counter,
        db=temp_db,
        session_id="proc_integration_session_01",
        snapshot_interval=5,
    )

    assert result.success is True
    assert result.session_id == "proc_integration_session_01"

    # Verify session was recorded and finalized in DB
    saved_session = temp_db.get_session("proc_integration_session_01")
    assert saved_session is not None
    assert saved_session["status"] == "completed"
    assert saved_session["duration_sec"] > 0
    assert saved_session["avg_fps"] > 0

    # Verify snapshots were recorded
    snaps = temp_db.get_snapshots_for_session("proc_integration_session_01")
    assert len(snaps) >= 2
