"""
ByteTrack Object Tracking Module for AI Traffic Monitoring System.

Integrates ByteTrack multi-object tracking via the supported Ultralytics API
to maintain persistent object identities, generate trajectory histories,
and support downstream line-crossing calculations.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import CLASS_COLORS_BGR, get_config
from core.detector import Detection, YOLODetector
from utils.logger import setup_logger

logger = setup_logger("tracker")


class ByteTracker:
    """
    ByteTrack-based multi-object tracker for traffic video analysis.
    """

    def __init__(
        self,
        detector: YOLODetector,
        tracker_type: Optional[str] = None,
        max_history: Optional[int] = None,
    ) -> None:
        """
        Initialize ByteTracker.

        Args:
            detector: Loaded YOLODetector instance.
            tracker_type: Tracker configuration file ('bytetrack.yaml').
            max_history: Maximum number of trajectory points to retain per track.
        """
        config = get_config()
        self.detector = detector
        self.tracker_type = tracker_type or config.tracker_type
        self.max_history = max_history or config.track_history_length

        # In-memory track history: track_id -> deque of (cx, cy)
        self.track_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        # Class memory for tracks (preserves class across momentary misclassifications)
        self.track_classes: Dict[int, str] = {}

    def track(self, frame: np.ndarray, track_all: bool = False) -> List[Detection]:
        """
        Process a video frame and update object tracks.

        Args:
            frame: Input BGR frame.
            track_all: If True, tracks all 80 COCO classes. If False, tracks traffic classes.

        Returns:
            List of structured Detection objects with valid track_id.
        """
        if not self.detector.is_ready():
            logger.error("Detector is not initialized for tracking.")
            return []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            logger.warning("Invalid or empty input frame provided to tracker.")
            return []

        target_class_ids = None if track_all else list(self.detector.target_classes.keys())

        try:
            # Run tracking with persist=True to maintain IDs across frames
            results = self.detector.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                conf=self.detector.confidence,
                iou=self.detector.iou,
                imgsz=self.detector.imgsz,
                device=self.detector.device,
                classes=target_class_ids,
                verbose=False,
            )
        except Exception as e:
            logger.error("Error during ByteTrack execution: %s", e)
            return []

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes = first_res.boxes.xyxy.cpu().numpy().astype(int)
        confidences = first_res.boxes.conf.cpu().numpy()
        class_ids = first_res.boxes.cls.cpu().numpy().astype(int)

        track_ids = (
            first_res.boxes.id.cpu().numpy().astype(int)
            if first_res.boxes.id is not None
            else [None] * len(boxes)
        )

        model_names = (
            self.detector.model.names
            if (self.detector.model is not None and hasattr(self.detector.model, "names"))
            else {}
        )

        for bbox, conf, cls_id, tid in zip(boxes, confidences, class_ids, track_ids):
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            raw_class_name = self.detector.target_classes.get(
                cls_id, model_names.get(cls_id, f"class_{cls_id}")
            )

            track_id_val = int(tid) if tid is not None else None

            # Stabilize class assignment using track history if available
            if track_id_val is not None:
                if track_id_val not in self.track_classes:
                    self.track_classes[track_id_val] = raw_class_name
                class_name = self.track_classes[track_id_val]
                # Append centroid to trajectory history
                self.track_history[track_id_val].append((cx, cy))
            else:
                class_name = raw_class_name

            detection = Detection(
                class_id=int(cls_id),
                class_name=class_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                track_id=track_id_val,
            )
            detections.append(detection)

        return detections

    def draw_tracks(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_trails: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, track labels, and trajectory history trails.

        Args:
            frame: Input BGR frame.
            detections: List of detections with tracking information.
            show_trails: Whether to render trailing motion history.
            show_confidence: Whether to include confidence score in label.

        Returns:
            Annotated frame.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        # First draw bounding boxes and badges via detector utility
        output = self.detector.draw_detections(
            frame, detections, show_track_id=True, show_confidence=show_confidence
        )

        if not show_trails:
            return output

        # Render trajectory motion trails
        for det in detections:
            if det.track_id is None:
                continue

            points = self.track_history.get(det.track_id)
            if not points or len(points) < 2:
                continue

            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            trail_color = CLASS_COLORS_BGR.get(det.class_name, (0, 255, 255))
            cv2.polylines(output, [pts], isClosed=False, color=trail_color, thickness=2)

        return output

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Return trajectory history coordinates for a specific track ID."""
        return list(self.track_history.get(track_id, []))

    def update_config(
        self,
        tracker_type: Optional[str] = None,
        max_history: Optional[int] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> None:
        """Adjust tracker and detector hyperparameters dynamically."""
        if tracker_type is not None:
            self.tracker_type = tracker_type
        if max_history is not None:
            self.max_history = max_history
        if confidence is not None:
            self.detector.confidence = confidence
        if iou is not None:
            self.detector.iou = iou

    def reset(self) -> None:
        """
        Reset tracker state and clear track histories for a new session.
        """
        self.track_history.clear()
        self.track_classes.clear()
        # Reset tracker in Ultralytics model if internal tracker exists
        try:
            if self.detector.model is not None and hasattr(self.detector.model, "predictor"):
                predictor = self.detector.model.predictor
                if predictor is not None and hasattr(predictor, "trackers"):
                    predictor.trackers = None
        except Exception as e:
            logger.debug("Notice on predictor tracker reset: %s", e)
        logger.info("ByteTracker history and tracking state reset.")
