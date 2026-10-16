"""
Vehicle Counting Module for AI Traffic Monitoring System.

Implements directional virtual line crossing using geometric side determination,
tracks unique observed vehicles by class, and prevents duplicate counts across frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config.settings import CLASS_COLORS_BGR, get_config
from core.detector import Detection
from utils.logger import setup_logger

logger = setup_logger("vehicle_counter")


@dataclass
class CountingLine:
    """Represents a virtual counting line segment in the frame coordinate space."""

    p1: Tuple[int, int]  # (x1, y1)
    p2: Tuple[int, int]  # (x2, y2)
    name: str = "Main Counting Line"

    def get_side(self, point: Tuple[int, int]) -> float:
        """
        Calculate signed 2D cross-product determinant to determine which side of the line
        the given point lies on.

        Returns:
            > 0: On one side (e.g., below / inbound)
            < 0: On the opposite side (e.g., above / outbound)
            0: Exactly colinear with the line
        """
        x1, y1 = self.p1
        x2, y2 = self.p2
        px, py = point
        return float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))

    def segments_intersect(
        self,
        p_prev: Tuple[int, int],
        p_curr: Tuple[int, int],
        buffer_px: int = 20,
    ) -> bool:
        """
        Check if the movement trajectory between p_prev and p_curr intersects
        this counting line segment.
        """
        # Side of line (p1 -> p2) for trajectory endpoints
        side_prev = self.get_side(p_prev)
        side_curr = self.get_side(p_curr)

        # If on the same side, no intersection
        if (side_prev * side_curr > 0):
            return False

        # Now check if line endpoints (p1, p2) are on opposite sides of trajectory line (p_prev -> p_curr)
        def side_of_trajectory(pt: Tuple[int, int]) -> float:
            tx1, ty1 = p_prev
            tx2, ty2 = p_curr
            return float((tx2 - tx1) * (pt[1] - ty1) - (ty2 - ty1) * (pt[0] - tx1))

        side_p1 = side_of_trajectory(self.p1)
        side_p2 = side_of_trajectory(self.p2)

        if (side_p1 * side_p2 > 0):
            return False

        return True


@dataclass
class CountingState:
    """Stores the cumulative counting state for a monitoring session."""

    total_inbound: int = 0
    total_outbound: int = 0
    class_counts: Dict[str, int] = field(
        default_factory=lambda: {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "person": 0}
    )
    unique_objects: Dict[int, str] = field(default_factory=dict)
    crossed_track_ids: Set[int] = field(default_factory=set)
    previous_positions: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    @property
    def total_crossed(self) -> int:
        return self.total_inbound + self.total_outbound

    @property
    def total_unique_vehicles(self) -> int:
        """Total unique motor vehicles (excluding pedestrians)."""
        return sum(
            1 for cls in self.unique_objects.values() if cls in ("car", "bus", "truck", "motorcycle")
        )

    @property
    def total_unique_pedestrians(self) -> int:
        return sum(1 for cls in self.unique_objects.values() if cls == "person")


class VehicleCounter:
    """
    Directional virtual line vehicle counter with duplicate prevention
    and class-wise aggregation.
    """

    def __init__(
        self,
        line_ratio: Optional[float] = None,
        custom_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> None:
        """
        Initialize VehicleCounter.

        Args:
            line_ratio: Vertical line position as a ratio of frame height (0.0 to 1.0).
            custom_line: Optional explicit coordinates ((x1, y1), (x2, y2)).
        """
        config = get_config()
        self.line_ratio = line_ratio if line_ratio is not None else config.default_line_ratio
        self.custom_line = custom_line
        self.line: Optional[CountingLine] = None

        self.state = CountingState()

    def update_line_geometry(self, frame_width: int, frame_height: int) -> None:
        """Update or initialize the virtual counting line based on frame dimensions."""
        if self.custom_line is not None:
            self.line = CountingLine(p1=self.custom_line[0], p2=self.custom_line[1])
        else:
            y_pos = int(frame_height * self.line_ratio)
            self.line = CountingLine(
                p1=(0, y_pos),
                p2=(frame_width, y_pos),
                name=f"Line (ratio={self.line_ratio:.2f})",
            )

    def process_frame(
        self,
        detections: List[Detection],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> CountingState:
        """
        Process a list of detections for the current frame, updating crossing and unique counts.

        Args:
            detections: List of current frame detections (with track_ids if available).
            frame_shape: Optional (height, width) to configure line geometry.

        Returns:
            Updated CountingState instance.
        """
        if frame_shape and (self.line is None):
            self.update_line_geometry(frame_shape[1], frame_shape[0])

        current_frame_track_ids = set()

        for det in detections:
            cls_name = det.class_name
            tid = det.track_id

            # If no track_id, cannot reliably count line crossing without tracking
            if tid is None:
                continue

            current_frame_track_ids.add(tid)

            # Record unique object observation
            if tid not in self.state.unique_objects:
                self.state.unique_objects[tid] = cls_name
                if cls_name in self.state.class_counts:
                    self.state.class_counts[cls_name] += 1
                else:
                    self.state.class_counts[cls_name] = 1

            # Check line crossing if counting line is defined
            if self.line is not None:
                # Use bottom_center for vehicle ground contact or centroid
                curr_point = det.bottom_center
                prev_point = self.state.previous_positions.get(tid)

                if prev_point is not None and tid not in self.state.crossed_track_ids:
                    # Check if line was crossed between prev_point and curr_point
                    if self.line.segments_intersect(prev_point, curr_point):
                        # Determine direction:
                        # Moving downwards (prev_point[1] < curr_point[1]) -> Inbound
                        # Moving upwards (prev_point[1] > curr_point[1]) -> Outbound
                        if curr_point[1] > prev_point[1]:
                            direction = "inbound"
                            self.state.total_inbound += 1
                        else:
                            direction = "outbound"
                            self.state.total_outbound += 1

                        self.state.crossed_track_ids.add(tid)
                        logger.info(
                            "Object #%d (%s) crossed %s [%s]",
                            tid,
                            cls_name,
                            self.line.name,
                            direction,
                        )

                # Update recorded position
                self.state.previous_positions[tid] = curr_point

        # Clean up stale tracks from previous_positions to conserve memory
        if len(self.state.previous_positions) > 1000:
            stale_keys = [
                k for k in self.state.previous_positions if k not in current_frame_track_ids
            ]
            for k in stale_keys[:200]:
                self.state.previous_positions.pop(k, None)

        return self.state

    def draw_counting_overlay(
        self,
        frame: np.ndarray,
        show_line: bool = True,
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Render virtual counting line and live counts HUD onto the frame.

        Args:
            frame: Input BGR frame.
            show_line: Whether to draw the virtual line.
            show_hud: Whether to render the count HUD panel.

        Returns:
            Annotated frame.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        if self.line is None:
            self.update_line_geometry(w, h)

        if self.line is not None and show_line:
            # Draw primary counting line (bright cyan/yellow)
            p1 = self.line.p1
            p2 = self.line.p2
            cv2.line(output, p1, p2, (0, 255, 255), 3)

            # Draw end handles
            cv2.circle(output, p1, 6, (0, 200, 255), -1)
            cv2.circle(output, p2, 6, (0, 200, 255), -1)

            # Line label
            label_x = min(p1[0] + 15, w - 240)
            label_y = max(p1[1] - 8, 20)
            cv2.putText(
                output,
                "COUNTING LINE",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if show_hud:
            # Semi-transparent HUD box in top-left
            hud_w, hud_h = 240, 115
            hud_bg = output[10 : 10 + hud_h, 10 : 10 + hud_w]
            overlay = np.zeros_like(hud_bg)
            cv2.addWeighted(hud_bg, 0.4, overlay, 0.6, 0, hud_bg)

            # HUD Border
            cv2.rectangle(output, (10, 10), (10 + hud_w, 10 + hud_h), (0, 255, 255), 1)

            # HUD Text
            cv2.putText(
                output,
                "TRAFFIC FLOW COUNTS",
                (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"Inbound:   {self.state.total_inbound}",
                (18, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"Outbound:  {self.state.total_outbound}",
                (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"Total Line Crossings: {self.state.total_crossed}",
                (18, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"Unique Observed: {len(self.state.unique_objects)}",
                (18, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return output

    def get_summary(self) -> Dict[str, Any]:
        """Return structured summary of current counts and metrics."""
        return {
            "total_inbound": self.state.total_inbound,
            "total_outbound": self.state.total_outbound,
            "total_crossed": self.state.total_crossed,
            "unique_vehicles": self.state.total_unique_vehicles,
            "unique_pedestrians": self.state.total_unique_pedestrians,
            "cars": self.state.class_counts.get("car", 0),
            "buses": self.state.class_counts.get("bus", 0),
            "trucks": self.state.class_counts.get("truck", 0),
            "motorcycles": self.state.class_counts.get("motorcycle", 0),
            "pedestrians": self.state.class_counts.get("person", 0),
            "class_counts": dict(self.state.class_counts),
        }

    def reset(self) -> None:
        """Reset all counters, crossing sets, and position states."""
        self.state = CountingState()
        logger.info("VehicleCounter state reset.")
