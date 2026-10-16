"""
Unit tests for vehicle counting and line-crossing logic.
"""

from __future__ import annotations

import pytest

from core.detector import Detection
from core.vehicle_counter import CountingLine, VehicleCounter


def create_detection(track_id: int, class_name: str, y_bottom: int, x_center: int = 320) -> Detection:
    """Helper to synthesize a Detection object with specific bottom-center coordinates."""
    w, h = 60, 80
    x1 = x_center - (w // 2)
    x2 = x_center + (w // 2)
    y1 = y_bottom - h
    y2 = y_bottom
    return Detection(
        class_id=2 if class_name == "car" else (0 if class_name == "person" else 5),
        class_name=class_name,
        confidence=0.88,
        bbox=(x1, y1, x2, y2),
        center=(x_center, (y1 + y2) // 2),
        track_id=track_id,
    )


def test_counting_line_geometry():
    """Test geometric side determination."""
    # Horizontal line across y=300 from x=0 to x=640
    line = CountingLine(p1=(0, 300), p2=(640, 300))

    # Point above line (y=250)
    side_above = line.get_side((320, 250))
    # Point below line (y=350)
    side_below = line.get_side((320, 350))

    assert side_above * side_below < 0, "Points on opposite sides must have opposite determinant signs"


def test_vehicle_remaining_on_one_side():
    """A vehicle moving on one side of the line should NOT count as crossing."""
    counter = VehicleCounter(line_ratio=0.50)
    # Frame size 640x480 -> line at y=240
    counter.update_line_geometry(640, 480)

    # Frame 1: Vehicle 1 at y=100
    f1_det = [create_detection(track_id=1, class_name="car", y_bottom=100)]
    counter.process_frame(f1_det)

    # Frame 2: Vehicle 1 moves to y=150 (still above line at 240)
    f2_det = [create_detection(track_id=1, class_name="car", y_bottom=150)]
    counter.process_frame(f2_det)

    summary = counter.get_summary()
    assert summary["total_crossed"] == 0
    assert summary["total_inbound"] == 0
    assert summary["total_outbound"] == 0
    assert summary["unique_vehicles"] == 1
    assert summary["cars"] == 1


def test_single_crossing_inbound():
    """A vehicle crossing from above line to below line is counted as inbound exactly once."""
    counter = VehicleCounter(line_ratio=0.50)
    counter.update_line_geometry(640, 480)  # line at y=240

    # Frame 1: above line at y=200
    counter.process_frame([create_detection(track_id=10, class_name="car", y_bottom=200)])
    assert counter.state.total_crossed == 0

    # Frame 2: crosses downwards to y=270 (inbound)
    counter.process_frame([create_detection(track_id=10, class_name="car", y_bottom=270)])
    assert counter.state.total_crossed == 1
    assert counter.state.total_inbound == 1
    assert counter.state.total_outbound == 0

    # Frame 3: continues moving downwards to y=320 -> should NOT count again!
    counter.process_frame([create_detection(track_id=10, class_name="car", y_bottom=320)])
    assert counter.state.total_crossed == 1
    assert counter.state.total_inbound == 1


def test_single_crossing_outbound():
    """A vehicle crossing upwards from below line to above line is counted as outbound."""
    counter = VehicleCounter(line_ratio=0.50)
    counter.update_line_geometry(640, 480)  # line at y=240

    # Frame 1: below line at y=310
    counter.process_frame([create_detection(track_id=20, class_name="bus", y_bottom=310)])
    assert counter.state.total_crossed == 0

    # Frame 2: crosses upwards to y=190 (outbound)
    counter.process_frame([create_detection(track_id=20, class_name="bus", y_bottom=190)])
    assert counter.state.total_crossed == 1
    assert counter.state.total_inbound == 0
    assert counter.state.total_outbound == 1


def test_missing_track_id_handling():
    """Detections without track IDs should not raise exceptions or trigger line crossings."""
    counter = VehicleCounter(line_ratio=0.50)
    counter.update_line_geometry(640, 480)

    det_no_id = Detection(
        class_id=2,
        class_name="car",
        confidence=0.85,
        bbox=(200, 200, 260, 280),
        center=(230, 240),
        track_id=None,
    )
    counter.process_frame([det_no_id])
    assert counter.state.total_crossed == 0
    assert len(counter.state.unique_objects) == 0


def test_multiple_vehicles_and_classes():
    """Test counting multiple vehicles of different classes."""
    counter = VehicleCounter(line_ratio=0.50)
    counter.update_line_geometry(640, 480)  # line at y=240

    # Frame 1: Car, Bus, Pedestrian approaching
    counter.process_frame([
        create_detection(track_id=1, class_name="car", y_bottom=210, x_center=150),
        create_detection(track_id=2, class_name="bus", y_bottom=220, x_center=300),
        create_detection(track_id=3, class_name="person", y_bottom=230, x_center=450),
    ])
    assert counter.state.total_crossed == 0

    # Frame 2: Car and Bus cross downwards, Pedestrian stops
    counter.process_frame([
        create_detection(track_id=1, class_name="car", y_bottom=260, x_center=150),
        create_detection(track_id=2, class_name="bus", y_bottom=270, x_center=300),
        create_detection(track_id=3, class_name="person", y_bottom=235, x_center=450),
    ])
    assert counter.state.total_crossed == 2
    assert counter.state.total_inbound == 2
    assert counter.state.class_counts["car"] == 1
    assert counter.state.class_counts["bus"] == 1
    assert counter.state.class_counts["person"] == 1

    summary = counter.get_summary()
    assert summary["unique_vehicles"] == 2
    assert summary["unique_pedestrians"] == 1


def test_counter_reset():
    """Test that resetting clears all counting state."""
    counter = VehicleCounter(line_ratio=0.50)
    counter.update_line_geometry(640, 480)

    counter.process_frame([create_detection(1, "car", 210)])
    counter.process_frame([create_detection(1, "car", 270)])
    assert counter.state.total_crossed == 1

    counter.reset()
    assert counter.state.total_crossed == 0
    assert counter.state.total_inbound == 0
    assert len(counter.state.unique_objects) == 0
