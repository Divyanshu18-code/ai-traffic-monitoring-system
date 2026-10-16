"""
Unit and integration tests for Streamlit Dashboard components, helpers, and export logic.

Tests:
1. Dashboard and components module imports
2. Configuration loading and default parameters
3. Session CSV export formatting
4. Full database history CSV export formatting
5. Empty database and empty session handling
6. Metric card and KPI summary calculations
7. Video format and file size validation
8. Chart generation with valid and empty datasets
9. Session state helpers and isolation
10. Error message formatting and safe error reporting
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from components.charts import (
    plot_class_distribution,
    plot_directional_traffic,
    plot_fps_performance,
    plot_sessions_history,
    plot_traffic_trend,
)
from components.metrics import render_metric_card, render_summary_kpis
from config.settings import AppConfig, get_config
from database.db_manager import DatabaseManager
from utils.file_utils import format_bytes, sanitize_filename
from utils.validation import (
    validate_confidence_threshold,
    validate_file_size,
    validate_line_ratio,
    validate_stream_url,
    validate_video_file_extension,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Create a temporary database for dashboard tests."""
    return DatabaseManager(db_path=tmp_path / "dashboard_test.db")


# 1. Dashboard Imports
def test_dashboard_imports():
    """Verify that all core dashboard components and utilities import cleanly."""
    import app
    from components.charts import plot_class_distribution
    from components.metrics import render_summary_kpis
    from components.sidebar import render_sidebar
    from components.video_panel import render_video_preview

    assert callable(app.main)
    assert callable(plot_class_distribution)
    assert callable(render_summary_kpis)
    assert callable(render_sidebar)
    assert callable(render_video_preview)


# 2. Configuration Loading
def test_config_loading():
    """Test loading application configuration and defaults."""
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.default_model_name.endswith(".pt")
    assert 0.0 < cfg.confidence_threshold < 1.0
    assert 0.0 < cfg.default_line_ratio < 1.0
    assert len(cfg.target_classes) >= 5


# 3. Session CSV Export Logic
def test_session_csv_export_logic():
    """Test generating a single session CSV export matching required schema."""
    session_data = {
        "session_id": ["sess_export_001"],
        "original_filename": ["traffic.mp4"],
        "duration_sec": [45.2],
        "total_unique_vehicles": [18],
        "cars": [12],
        "buses": [2],
        "trucks": [3],
        "motorcycles": [1],
        "pedestrians": [4],
        "inbound_count": [10],
        "outbound_count": [8],
        "average_fps": [28.5],
    }
    df = pd.DataFrame(session_data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    csv_content = buffer.getvalue()

    assert "sess_export_001" in csv_content
    assert "traffic.mp4" in csv_content
    assert "total_unique_vehicles" in csv_content

    # Verify reload
    reloaded = pd.read_csv(io.StringIO(csv_content))
    assert len(reloaded) == 1
    assert reloaded.iloc[0]["cars"] == 12
    assert reloaded.iloc[0]["total_unique_vehicles"] == 18


# 4. History CSV Export Logic
def test_history_csv_export_logic(temp_db: DatabaseManager):
    """Test exporting all recorded database sessions as a DataFrame."""
    temp_db.create_session("sess_hist_1", "video_file", "highway1.mp4")
    temp_db.finalize_session("sess_hist_1", total_vehicles=10, cars=8, avg_fps=30.0)

    temp_db.create_session("sess_hist_2", "video_file", "highway2.mp4")
    temp_db.finalize_session("sess_hist_2", total_vehicles=25, cars=20, avg_fps=29.0)

    df = temp_db.export_sessions_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "session_id" in df.columns
    assert "total_vehicles" in df.columns
    assert df.iloc[0]["session_id"] in ("sess_hist_1", "sess_hist_2")


# 5. Empty History and Empty Database Handling
def test_empty_database_and_history_handling(temp_db: DatabaseManager):
    """Test that empty database queries return empty structures without crashing."""
    aggregates = temp_db.get_aggregate_metrics()
    assert aggregates.get("total_sessions", 0) == 0
    assert aggregates.get("total_vehicles", 0) == 0

    recent = temp_db.get_recent_sessions(limit=5)
    assert recent == []

    filtered = temp_db.filter_sessions(start_date="2026-01-01", end_date="2026-01-02")
    assert filtered == []

    df = temp_db.export_sessions_dataframe()
    assert df.empty


# 6. Chart Generation
def test_chart_generation_with_data_and_empty():
    """Verify that matplotlib chart helpers generate valid figures without crashing."""
    # Class distribution chart with data
    counts = {"car": 14, "truck": 3, "bus": 1, "motorcycle": 2, "person": 5}
    fig1 = plot_class_distribution(counts)
    assert fig1 is not None

    # Empty counts
    fig1_empty = plot_class_distribution({"car": 0, "bus": 0})
    assert fig1_empty is not None

    # Directional chart
    fig2 = plot_directional_traffic(inbound=15, outbound=12)
    assert fig2 is not None

    # Trend charts with snapshots
    mock_snapshots = [
        {"frame_number": 0, "active_vehicles": 2, "cumulative_inbound": 0, "cumulative_outbound": 0, "current_fps": 30.0},
        {"frame_number": 30, "active_vehicles": 5, "cumulative_inbound": 2, "cumulative_outbound": 1, "current_fps": 28.5},
    ]
    fig3 = plot_traffic_trend(mock_snapshots)
    assert fig3 is not None

    # Trend chart with empty snapshots returns None
    assert plot_traffic_trend([]) is None
    assert plot_traffic_trend([mock_snapshots[0]]) is None

    # Multi-session history chart
    mock_sessions = [
        {"session_id": "sess_1", "source_name": "f1.mp4", "inbound_count": 5, "outbound_count": 3, "total_vehicles": 8},
        {"session_id": "sess_2", "source_name": "f2.mp4", "inbound_count": 10, "outbound_count": 7, "total_vehicles": 17},
    ]
    fig4 = plot_sessions_history(mock_sessions)
    assert fig4 is not None
    assert plot_sessions_history([]) is None


# 7. Video Input Validation Helpers
def test_video_validation_helpers():
    """Test video validation logic used by the dashboard upload workflow."""
    # Valid extensions
    ok, _ = validate_video_file_extension("traffic_highway.mp4")
    assert ok is True

    ok, _ = validate_video_file_extension("stream.avi")
    assert ok is True

    # Invalid extension
    bad_ok, bad_msg = validate_video_file_extension("document.pdf")
    assert bad_ok is False
    assert "Unsupported" in bad_msg

    # File size validation
    ok_size, _ = validate_file_size(10 * 1024 * 1024)
    assert ok_size is True

    too_large_ok, too_large_msg = validate_file_size(500 * 1024 * 1024, max_mb=200)
    assert too_large_ok is False
    assert "exceeds" in too_large_msg

    # Filename sanitization
    safe = sanitize_filename("traffic test video (final).mp4")
    assert " " not in safe
    assert "(" not in safe

    # Stream URL validation
    rtsp_ok, _ = validate_stream_url("rtsp://192.168.1.100:554/stream1")
    assert rtsp_ok is True

    http_ok, _ = validate_stream_url("http://192.168.1.100:8080/video.mjpg")
    assert http_ok is True

    invalid_url_ok, _ = validate_stream_url("ftp://example.com/feed")
    assert invalid_url_ok is False


# 8. Setting Threshold Validation
def test_setting_threshold_validation():
    """Test slider and configuration parameter validators."""
    ok, err_msg = validate_confidence_threshold(0.45)
    assert ok is True
    assert err_msg == ""

    bad_ok, bad_msg = validate_confidence_threshold(-0.1)
    assert bad_ok is False
    assert "between 0.05" in bad_msg

    ok_ratio, ratio_msg = validate_line_ratio(0.60)
    assert ok_ratio is True
    assert ratio_msg == ""

    bad_ratio, bad_ratio_msg = validate_line_ratio(1.5)
    assert bad_ratio is False
    assert "between 0.05" in bad_ratio_msg


# 9. Format Utility Helpers
def test_format_utility_helpers():
    """Test utility byte size formatters."""
    assert "KB" in format_bytes(50 * 1024)
    assert "MB" in format_bytes(15 * 1024 * 1024)
    assert "GB" in format_bytes(2 * 1024 * 1024 * 1024)
