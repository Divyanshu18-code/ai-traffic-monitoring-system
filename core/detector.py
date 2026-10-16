"""
YOLO Object Detector Module for AI Traffic Monitoring System.

Provides robust object detection for vehicles (cars, buses, trucks, motorcycles)
and pedestrians using Ultralytics YOLO with configurable parameters,
CPU/CUDA auto-detection, structured outputs, and overlay rendering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    CLASS_COLORS_BGR,
    COCO_TRAFFIC_CLASSES,
    MODELS_DIR,
    PROJECT_ROOT,
    get_config,
)
from utils.logger import setup_logger

logger = setup_logger("detector")


@dataclass
class Detection:
    """Structured representation of a single detected or tracked object."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (center_x, center_y)
    track_id: Optional[int] = None

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Bottom-center point of bounding box, ideal for ground-contact line crossing."""
        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int(self.bbox[3])
        return cx, cy


class YOLODetector:
    """
    YOLOv8 Object Detector for traffic surveillance and monitoring.
    """

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the YOLO object detector.

        Args:
            model_name_or_path: Path or standard name (e.g., 'yolov8n.pt').
            confidence: Detection confidence threshold (0.0 to 1.0).
            iou: IoU threshold for NMS.
            imgsz: Inference image size.
            device: 'cpu', 'cuda', or '0'. If None, detects automatically.
        """
        config = get_config()
        self.confidence = confidence if confidence is not None else config.confidence_threshold
        self.iou = iou if iou is not None else config.iou_threshold
        self.imgsz = imgsz if imgsz is not None else config.imgsz
        self.target_classes = config.target_classes

        # Resolve compute device
        self.device = self._resolve_device(device or config.device)

        # Resolve model weights path
        self.model_path = self._resolve_model_path(model_name_or_path or config.default_model_name)

        # Load YOLO model
        self.model: Optional[YOLO] = None
        self._load_model()

    def _resolve_device(self, requested_device: str) -> str:
        """Verify device availability with clear CPU fallback."""
        if not requested_device or not isinstance(requested_device, str):
            return "cpu"
        requested = requested_device.lower().strip()
        if requested in ("cuda", "0", "gpu"):
            if torch.cuda.is_available():
                logger.info("CUDA GPU acceleration enabled: %s", torch.cuda.get_device_name(0))
                return "cuda"
            else:
                logger.warning("CUDA requested but not available on this machine. Falling back to CPU.")
                return "cpu"
        return "cpu"

    def _resolve_model_path(self, model_input: Union[str, Path]) -> str:
        """Locate weights in models/ folder, local workspace, or resolve standard name."""
        if not model_input or not str(model_input).strip():
            raise ValueError("Model path or name cannot be empty.")

        input_path = Path(model_input)

        # 1. Direct path exists
        if input_path.is_file():
            return str(input_path.resolve())

        # 2. Check in models/ directory
        candidate_in_models = MODELS_DIR / input_path.name
        if candidate_in_models.is_file():
            return str(candidate_in_models.resolve())

        # 3. Check in project root
        candidate_in_root = PROJECT_ROOT / input_path.name
        if candidate_in_root.is_file():
            return str(candidate_in_root.resolve())

        # 4. Standard YOLO model name (Ultralytics handles standard download if network available)
        name_lower = input_path.name.lower()
        if (name_lower.startswith("yolo") or name_lower.startswith("rtdetr")) and name_lower.endswith(".pt"):
            return str(input_path.name)

        # 5. Otherwise raise FileNotFoundError
        raise FileNotFoundError(
            f"Model weights file '{model_input}' not found in '{MODELS_DIR}' or project root."
        )

    def _load_model(self) -> None:
        """Load YOLO model weights with error handling."""
        try:
            logger.info("Loading YOLO model: %s on device: %s", self.model_path, self.device)
            self.model = YOLO(self.model_path)
            logger.info("YOLO model successfully initialized.")
        except FileNotFoundError as fnf:
            logger.error("Model file not found: %s", fnf)
            self.model = None
            raise
        except Exception as e:
            logger.error("Failed to initialize or download YOLO model '%s': %s", self.model_path, e)
            self.model = None
            raise RuntimeError(
                f"Could not load or download YOLO model weights from '{self.model_path}': {e}"
            ) from e

    def is_ready(self) -> bool:
        """Check if detector is ready for inference."""
        return self.model is not None

    def detect(self, frame: np.ndarray, detect_all: bool = False) -> List[Detection]:
        """
        Run object detection on an individual video frame.

        Args:
            frame: BGR numpy image frame.
            detect_all: If True, detects all 80 COCO classes (e.g. for general webcam vision).
                       If False, filters strictly to traffic classes.

        Returns:
            List of structured Detection objects.
        """
        if not self.is_ready():
            logger.error("Detector model is not loaded.")
            return []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            logger.warning("Invalid or empty input frame provided to detector.")
            return []

        target_class_ids = None if detect_all else list(self.target_classes.keys())

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                classes=target_class_ids,
                verbose=False,
            )
        except Exception as e:
            logger.error("Error during YOLO inference: %s", e)
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

        model_names = self.model.names if hasattr(self.model, "names") else {}

        for bbox, conf, cls_id, tid in zip(boxes, confidences, class_ids, track_ids):
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cls_name = self.target_classes.get(cls_id, model_names.get(cls_id, f"class_{cls_id}"))

            detection = Detection(
                class_id=int(cls_id),
                class_name=cls_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                track_id=int(tid) if tid is not None else None,
            )
            detections.append(detection)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_track_id: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and tracking IDs onto the frame.

        Args:
            frame: Input BGR frame.
            detections: List of detections to draw.
            show_track_id: Whether to display tracking ID.
            show_confidence: Whether to display confidence score.

        Returns:
            Annotated frame (modified in-place or copied).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            logger.warning("Invalid or empty frame provided for drawing detections.")
            return frame

        output = frame.copy()
        if not detections:
            return output

        h, w = output.shape[:2]

        palette = [
            (0, 210, 106),   # green
            (59, 130, 246),  # blue
            (245, 158, 11),  # orange
            (139, 92, 246),  # purple
            (236, 72, 153),  # pink
            (20, 184, 166),  # teal
            (244, 63, 94),   # red
            (6, 182, 212),   # cyan
            (234, 179, 8),   # yellow
        ]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            cx = max(0, min(det.center[0], w - 1))
            cy = max(0, min(det.center[1], h - 1))

            if det.class_name in CLASS_COLORS_BGR:
                color = CLASS_COLORS_BGR[det.class_name]
            else:
                color = palette[det.class_id % len(palette)]

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cv2.circle(output, (cx, cy), 3, color, -1)

            # Prepare label text
            label_parts = [det.class_name.capitalize()]
            if show_track_id and det.track_id is not None:
                label_parts.insert(0, f"#{det.track_id}")
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")

            label_text = " ".join(label_parts)

            # Draw label badge with background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(output.shape[1], x1 + text_w + 8)

            cv2.rectangle(output, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                output,
                label_text,
                (badge_x1 + 4, badge_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return output

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence,
            "iou_threshold": self.iou,
            "imgsz": self.imgsz,
            "is_ready": self.is_ready(),
            "target_classes": self.target_classes,
        }
