"""
Application Settings and Configuration Module.

Provides centralized configuration management with environment variable overrides
and default fallback options for model, processing, and UI settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
for directory in [MODELS_DIR, DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, SAMPLES_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Standard COCO Class Mapping for Traffic Monitoring
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_TRAFFIC_CLASSES: Dict[int, str] = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color mapping in BGR for OpenCV visualization
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 128, 0),       # Blue-Orange
    "car": (0, 255, 0),             # Green
    "motorcycle": (255, 255, 0),    # Cyan
    "bus": (0, 165, 255),           # Orange
    "truck": (0, 0, 255),           # Red
}

# Color mapping in HEX for Streamlit charts
CLASS_COLORS_HEX: Dict[str, str] = {
    "person": "#0080FF",
    "car": "#00FF66",
    "motorcycle": "#00FFFF",
    "bus": "#FFA500",
    "truck": "#FF3333",
}


@dataclass
class AppConfig:
    """Central configuration class for AI Traffic Monitoring System."""

    # Project Information
    app_name: str = "AI-Powered Traffic Monitoring & Analytics"
    version: str = "1.0.0"
    debug: bool = False

    # Directories
    project_root: Path = PROJECT_ROOT
    models_dir: Path = MODELS_DIR
    uploads_dir: Path = UPLOADS_DIR
    processed_dir: Path = PROCESSED_DIR
    samples_dir: Path = SAMPLES_DIR
    logs_dir: Path = LOGS_DIR
    reports_dir: Path = REPORTS_DIR

    # YOLO Model Settings
    default_model_name: str = "yolov8n.pt"
    available_models: List[str] = field(
        default_factory=lambda: ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    )
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"  # 'cpu', 'cuda', or '0' if GPU available

    # Tracking Settings (ByteTrack)
    tracker_type: str = "bytetrack.yaml"
    track_history_length: int = 30

    # Target Detection Classes
    target_classes: Dict[int, str] = field(default_factory=lambda: dict(COCO_TRAFFIC_CLASSES))

    # Video Processing
    allowed_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    max_upload_size_mb: int = 200
    default_fps: float = 30.0
    display_width: int = 960

    # Counting Line Settings
    # Line ratio relative to frame height (0.0 top, 1.0 bottom)
    default_line_ratio: float = 0.60
    counting_line_offset_pixels: int = 20  # buffer band for crossing detection

    # Congestion Heuristics Thresholds (Vehicles present simultaneously)
    congestion_low_threshold: int = 5
    congestion_moderate_threshold: int = 15

    # Database
    db_path: Path = DATA_DIR / "traffic_analytics.db"

    # Logging
    log_file: Path = LOGS_DIR / "traffic_system.log"
    log_level: str = "INFO"


_global_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Retrieve or initialize the singleton application configuration."""
    global _global_config
    if _global_config is None:
        _global_config = AppConfig(
            debug=os.getenv("APP_DEBUG", "false").lower() in ("true", "1"),
            default_model_name=os.getenv("DEFAULT_MODEL_NAME", "yolov8n.pt"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
            device=os.getenv("DEVICE", "cpu"),
            db_path=Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "traffic_analytics.db"))),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    return _global_config
