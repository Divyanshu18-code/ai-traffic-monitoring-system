"""
Traffic Analytics Module for AI Traffic Monitoring System.

Calculates real-time traffic density, heuristic congestion classification,
vehicle class distribution, FPS processing throughput, and temporal snapshots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_config
from core.detector import Detection
from core.vehicle_counter import CountingState
from utils.logger import setup_logger

logger = setup_logger("traffic_analyzer")


@dataclass
class TrafficSnapshot:
    """Represents a momentary snapshot of traffic status at a specific frame."""

    timestamp: float
    frame_number: int
    active_vehicles: int
    active_pedestrians: int
    density_ratio: float
    congestion_level: str  # 'LOW', 'MODERATE', 'HIGH'
    current_fps: float
    cumulative_inbound: int
    cumulative_outbound: int


class TrafficAnalyzer:
    """
    Computes analytical metrics, vehicle density, and congestion heuristics
    from live frame detections and counting events.
    """

    def __init__(
        self,
        low_threshold: Optional[int] = None,
        moderate_threshold: Optional[int] = None,
    ) -> None:
        """
        Initialize TrafficAnalyzer.

        Args:
            low_threshold: Max active vehicles for 'LOW' congestion (heuristic).
            moderate_threshold: Max active vehicles for 'MODERATE' congestion (heuristic).
        """
        config = get_config()
        self.low_threshold = (
            low_threshold if low_threshold is not None else config.congestion_low_threshold
        )
        self.moderate_threshold = (
            moderate_threshold
            if moderate_threshold is not None
            else config.congestion_moderate_threshold
        )

        self.snapshots: List[TrafficSnapshot] = []
        self.start_time: float = time.time()
        self.total_frames_processed: int = 0
        self.fps_measurements: List[float] = []

    def classify_congestion(self, active_vehicle_count: int, density_ratio: float = 0.0) -> str:
        """
        Heuristic congestion classification.

        Note: Thresholds are operational heuristics and configurable.
        - LOW: active vehicles < low_threshold and density < 0.15
        - MODERATE: low_threshold <= active vehicles <= moderate_threshold
        - HIGH: active vehicles > moderate_threshold or density > 0.35

        Args:
            active_vehicle_count: Motor vehicles in current frame.
            density_ratio: Bounding box area over frame area.

        Returns:
            'LOW', 'MODERATE', or 'HIGH'.
        """
        if active_vehicle_count > self.moderate_threshold or density_ratio > 0.35:
            return "HIGH"
        elif active_vehicle_count >= self.low_threshold or density_ratio > 0.15:
            return "MODERATE"
        return "LOW"

    def calculate_density(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> float:
        """
        Estimate vehicle surface density as the ratio of bounding box area
        to total frame area.

        Args:
            detections: List of detections in current frame.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Float density ratio between 0.0 and 1.0.
        """
        if frame_width <= 0 or frame_height <= 0:
            return 0.0

        total_frame_area = frame_width * frame_height
        total_bbox_area = sum(
            det.width * det.height
            for det in detections
            if det.class_name in ("car", "bus", "truck", "motorcycle")
        )

        density = total_bbox_area / total_frame_area
        return round(min(1.0, max(0.0, density)), 4)

    def record_frame(
        self,
        frame_idx: int,
        detections: List[Detection],
        counting_state: CountingState,
        frame_shape: Tuple[int, int],
        frame_elapsed_sec: float,
    ) -> TrafficSnapshot:
        """
        Process frame results and record an analytical snapshot.

        Args:
            frame_idx: Current frame sequence index.
            detections: Detected objects in the frame.
            counting_state: Current cumulative counting state.
            frame_shape: (height, width) of the frame.
            frame_elapsed_sec: Time taken to process the frame in seconds.

        Returns:
            Recorded TrafficSnapshot.
        """
        self.total_frames_processed += 1

        # Calculate instantaneous FPS
        current_fps = 1.0 / frame_elapsed_sec if frame_elapsed_sec > 0 else 0.0
        self.fps_measurements.append(current_fps)
        if len(self.fps_measurements) > 100:
            self.fps_measurements.pop(0)

        # Count active objects in current frame
        active_vehicles = sum(
            1 for d in detections if d.class_name in ("car", "bus", "truck", "motorcycle")
        )
        active_pedestrians = sum(1 for d in detections if d.class_name == "person")

        # Density calculation
        h, w = frame_shape[:2]
        density = self.calculate_density(detections, w, h)

        # Congestion heuristic
        congestion = self.classify_congestion(active_vehicles, density)

        snapshot = TrafficSnapshot(
            timestamp=time.time(),
            frame_number=frame_idx,
            active_vehicles=active_vehicles,
            active_pedestrians=active_pedestrians,
            density_ratio=density,
            congestion_level=congestion,
            current_fps=round(current_fps, 1),
            cumulative_inbound=counting_state.total_inbound,
            cumulative_outbound=counting_state.total_outbound,
        )

        self.snapshots.append(snapshot)
        return snapshot

    @property
    def average_fps(self) -> float:
        """Calculate moving average processing FPS."""
        if not self.fps_measurements:
            return 0.0
        return round(sum(self.fps_measurements) / len(self.fps_measurements), 2)

    @property
    def session_duration_seconds(self) -> float:
        """Elapsed session monitoring duration in seconds."""
        return round(time.time() - self.start_time, 2)

    def get_class_distribution(
        self, counting_state: CountingState
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute category distribution percentages and counts.

        Returns:
            Dictionary mapping class name to {'count': int, 'percentage': float}.
        """
        counts = counting_state.class_counts
        total = sum(counts.values())

        dist = {}
        for cls, count in counts.items():
            pct = (count / total * 100.0) if total > 0 else 0.0
            dist[cls] = {"count": count, "percentage": round(pct, 1)}
        return dist

    def get_analytics_summary(self, counting_state: CountingState) -> Dict[str, Any]:
        """
        Generate complete analytical summary for reporting and dashboard cards.
        """
        class_dist = self.get_class_distribution(counting_state)
        latest_snapshot = self.snapshots[-1] if self.snapshots else None

        return {
            "session_duration_sec": self.session_duration_seconds,
            "total_frames_processed": self.total_frames_processed,
            "average_fps": self.average_fps,
            "total_unique_vehicles": counting_state.total_unique_vehicles,
            "total_unique_pedestrians": counting_state.total_unique_pedestrians,
            "total_line_crossings": counting_state.total_crossed,
            "inbound_count": counting_state.total_inbound,
            "outbound_count": counting_state.total_outbound,
            "current_congestion": latest_snapshot.congestion_level if latest_snapshot else "LOW",
            "current_density_ratio": latest_snapshot.density_ratio if latest_snapshot else 0.0,
            "active_vehicles": latest_snapshot.active_vehicles if latest_snapshot else 0,
            "class_distribution": class_dist,
            "snapshot_count": len(self.snapshots),
        }

    def reset(self) -> None:
        """Reset analytical buffers and timer."""
        self.snapshots.clear()
        self.fps_measurements.clear()
        self.total_frames_processed = 0
        self.start_time = time.time()
        logger.info("TrafficAnalyzer reset.")
