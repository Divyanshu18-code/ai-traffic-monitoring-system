"""
Unit tests for traffic analytics, density estimation, and congestion heuristics.
"""

from __future__ import annotations

import pytest

from core.detector import Detection
from core.traffic_analyzer import TrafficAnalyzer
from core.vehicle_counter import CountingState


def make_detection(class_name: str, width: int = 100, height: int = 100) -> Detection:
    """Helper to synthesize detection with specific bounding box dimensions."""
    return Detection(
        class_id=2 if class_name == "car" else (0 if class_name == "person" else 7),
        class_name=class_name,
        confidence=0.9,
        bbox=(0, 0, width, height),
        center=(width // 2, height // 2),
        track_id=1,
    )


def test_classify_congestion_thresholds():
    """Test heuristic congestion classification rules."""
    analyzer = TrafficAnalyzer(low_threshold=5, moderate_threshold=15)

    # Low: 3 vehicles, low density
    assert analyzer.classify_congestion(active_vehicle_count=3, density_ratio=0.05) == "LOW"

    # Moderate: 8 vehicles
    assert analyzer.classify_congestion(active_vehicle_count=8, density_ratio=0.10) == "MODERATE"

    # Moderate via density: 3 vehicles but high road coverage (e.g., 2 large buses)
    assert analyzer.classify_congestion(active_vehicle_count=3, density_ratio=0.20) == "MODERATE"

    # High: 18 vehicles
    assert analyzer.classify_congestion(active_vehicle_count=18, density_ratio=0.10) == "HIGH"

    # High via density
    assert analyzer.classify_congestion(active_vehicle_count=4, density_ratio=0.40) == "HIGH"


def test_density_calculation():
    """Test frame bounding box surface area density ratio."""
    analyzer = TrafficAnalyzer()

    # Frame 1000 x 1000 = 1,000,000 area
    # 2 cars of 200 x 200 = 40,000 + 40,000 = 80,000
    # Expected density = 80,000 / 1,000,000 = 0.08
    dets = [
        make_detection("car", 200, 200),
        make_detection("car", 200, 200),
        make_detection("person", 50, 100),  # Persons excluded from vehicle density
    ]

    density = analyzer.calculate_density(dets, frame_width=1000, frame_height=1000)
    assert density == 0.08

    # Empty detections -> 0.0
    assert analyzer.calculate_density([], 1000, 1000) == 0.0


def test_record_frame_and_fps():
    """Test recording frame snapshots and moving average FPS."""
    analyzer = TrafficAnalyzer()
    state = CountingState(total_inbound=4, total_outbound=2)
    dets = [make_detection("car"), make_detection("truck")]

    # Elapsed 0.05s -> ~20 FPS
    snap = analyzer.record_frame(
        frame_idx=1,
        detections=dets,
        counting_state=state,
        frame_shape=(480, 640),
        frame_elapsed_sec=0.05,
    )

    assert snap.frame_number == 1
    assert snap.active_vehicles == 2
    assert snap.active_pedestrians == 0
    assert snap.cumulative_inbound == 4
    assert snap.cumulative_outbound == 2
    assert snap.current_fps == 20.0
    assert len(analyzer.snapshots) == 1
    assert analyzer.average_fps == 20.0


def test_class_distribution():
    """Test category breakdown and percentage calculations."""
    analyzer = TrafficAnalyzer()
    state = CountingState(
        class_counts={"car": 6, "bus": 2, "truck": 2, "motorcycle": 0, "person": 0}
    )

    dist = analyzer.get_class_distribution(state)
    assert dist["car"]["count"] == 6
    assert dist["car"]["percentage"] == 60.0
    assert dist["bus"]["count"] == 2
    assert dist["bus"]["percentage"] == 20.0
    assert dist["truck"]["percentage"] == 20.0


def test_analytics_summary_and_reset():
    """Test comprehensive summary output and reset behavior."""
    analyzer = TrafficAnalyzer()
    state = CountingState(total_inbound=5, total_outbound=3)
    dets = [make_detection("bus")]
    analyzer.record_frame(1, dets, state, (480, 640), 0.04)

    summary = analyzer.get_analytics_summary(state)
    assert summary["total_frames_processed"] == 1
    assert summary["inbound_count"] == 5
    assert summary["outbound_count"] == 3
    assert summary["active_vehicles"] == 1
    assert "class_distribution" in summary

    analyzer.reset()
    assert analyzer.total_frames_processed == 0
    assert len(analyzer.snapshots) == 0
    assert analyzer.average_fps == 0.0
