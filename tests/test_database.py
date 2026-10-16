"""
Unit tests for SQLite database manager.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from database.db_manager import DatabaseManager


@pytest.fixture
def temp_db(tmp_path: Path):
    """Create a temporary SQLite database for testing using pytest tmp_path."""
    test_db_path = tmp_path / "test_traffic.db"
    db = DatabaseManager(db_path=test_db_path)
    return db


def test_database_initialization(temp_db):
    """Test that tables are created upon initialization."""
    assert temp_db.db_path.exists()
    all_sessions = temp_db.get_all_sessions()
    assert all_sessions == []


def test_session_creation_and_retrieval(temp_db):
    """Test creating and retrieving a session."""
    session_id = "test_sess_001"
    created = temp_db.create_session(
        session_id=session_id,
        source_type="video_file",
        source_name="traffic_sample.mp4",
        notes="Integration test session",
    )
    assert created is True

    # Duplicate session creation should return False
    dup_created = temp_db.create_session(
        session_id=session_id,
        source_type="video_file",
        source_name="traffic_sample.mp4",
    )
    assert dup_created is False

    session = temp_db.get_session(session_id)
    assert session is not None
    assert session["session_id"] == session_id
    assert session["source_type"] == "video_file"
    assert session["source_name"] == "traffic_sample.mp4"
    assert session["total_vehicles"] == 0


def test_save_snapshot(temp_db):
    """Test adding periodic traffic snapshots."""
    session_id = "test_sess_snapshot"
    temp_db.create_session(session_id, "webcam", "webcam_0")

    saved1 = temp_db.save_snapshot(
        session_id=session_id,
        frame_number=30,
        active_vehicles=4,
        congestion_level="LOW",
        cumulative_inbound=2,
        cumulative_outbound=1,
        current_fps=28.5,
    )
    assert saved1 is True

    saved2 = temp_db.save_snapshot(
        session_id=session_id,
        frame_number=60,
        active_vehicles=8,
        congestion_level="MODERATE",
        cumulative_inbound=5,
        cumulative_outbound=3,
        current_fps=29.1,
    )
    assert saved2 is True

    snapshots = temp_db.get_snapshots_for_session(session_id)
    assert len(snapshots) == 2
    assert snapshots[0]["frame_number"] == 30
    assert snapshots[1]["congestion_level"] == "MODERATE"


def test_finalize_session_and_aggregates(temp_db):
    """Test finalizing session and calculating aggregates."""
    session_id = "test_sess_finalize"
    temp_db.create_session(session_id, "video_file", "highway.mp4")

    finalized = temp_db.finalize_session(
        session_id=session_id,
        duration_sec=12.5,
        total_vehicles=20,
        cars=14,
        buses=2,
        trucks=3,
        motorcycles=1,
        pedestrians=4,
        inbound_count=12,
        outbound_count=8,
        avg_fps=27.4,
        output_video_path="data/processed/annotated.mp4",
    )
    assert finalized is True

    session = temp_db.get_session(session_id)
    assert session["total_vehicles"] == 20
    assert session["cars"] == 14
    assert session["buses"] == 2
    assert session["trucks"] == 3
    assert session["motorcycles"] == 1
    assert session["pedestrians"] == 4
    assert session["inbound_count"] == 12
    assert session["outbound_count"] == 8
    assert session["avg_fps"] == 27.4

    aggregates = temp_db.get_aggregate_metrics()
    assert aggregates["total_sessions"] == 1
    assert aggregates["total_vehicles"] == 20
    assert aggregates["total_cars"] == 14
    assert aggregates["total_inbound"] == 12


def test_date_and_source_filtering(temp_db):
    """Test filtering sessions by source type and date."""
    now = datetime.now()
    yesterday = now - timedelta(days=1)

    temp_db.create_session("sess_today_file", "video_file", "file1.mp4", start_time=now)
    temp_db.create_session("sess_today_cam", "webcam", "cam0", start_time=now)
    temp_db.create_session("sess_yest_file", "video_file", "file2.mp4", start_time=yesterday)

    today_str = now.strftime("%Y-%m-%d")
    today_sessions = temp_db.filter_sessions(start_date=today_str, end_date=today_str)
    assert len(today_sessions) == 2

    webcam_sessions = temp_db.filter_sessions(source_type="webcam")
    assert len(webcam_sessions) == 1
    assert webcam_sessions[0]["session_id"] == "sess_today_cam"


def test_export_sessions_dataframe(temp_db):
    """Test exporting session records as Pandas DataFrame."""
    temp_db.create_session("sess_df_test", "video_file", "test.mp4")
    df = temp_db.export_sessions_dataframe()
    assert not df.empty
    assert "session_id" in df.columns
    assert df.iloc[0]["session_id"] == "sess_df_test"


def test_delete_session(temp_db):
    """Test deleting a session and cascade integrity."""
    session_id = "sess_to_delete"
    temp_db.create_session(session_id, "video_file", "test.mp4")
    temp_db.save_snapshot(session_id, 1, 3)

    deleted = temp_db.delete_session(session_id)
    assert deleted is True
    assert temp_db.get_session(session_id) is None
    assert len(temp_db.get_snapshots_for_session(session_id)) == 0
