"""
Video utility helpers for AI Traffic Monitoring System.

Includes synthetic test video generator for offline testing and verification,
and video encoding helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import CLASS_COLORS_BGR, get_config
from utils.logger import setup_logger

logger = setup_logger("video_utils")


def generate_synthetic_traffic_video(
    output_path: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    duration_sec: int = 5,
) -> Path:
    """
    Generate a synthetic test traffic video with simulated moving vehicles
    crossing a central virtual line.

    Useful for testing video pipeline, line crossing, and analytics
    without requiring internet access or external video downloads.

    Args:
        output_path: Target path for the video file.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        duration_sec: Video length in seconds.

    Returns:
        Path to the generated video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        # Fallback to XVID / AVI if mp4v codec is unavailable
        output_path = output_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = fps * duration_sec

    # Simulated objects moving downwards (inbound) and upwards (outbound)
    # [x, start_y, speed, width, height, color_bgr, label]
    objects = [
        {"x": 160, "y": 20, "speed": 4, "w": 60, "h": 90, "color": (0, 255, 0), "label": "Car 1"},
        {"x": 260, "y": 60, "speed": 3, "w": 80, "h": 120, "color": (0, 165, 255), "label": "Bus 1"},
        {"x": 400, "y": height - 40, "speed": -5, "w": 60, "h": 90, "color": (0, 255, 0), "label": "Car 2"},
        {"x": 500, "y": height - 20, "speed": -3, "w": 50, "h": 40, "color": (255, 255, 0), "label": "Motorcycle 1"},
    ]

    for frame_idx in range(total_frames):
        # Create road asphalt background
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # Draw road lane markings
        cv2.line(frame, (width // 2, 0), (width // 2, height), (220, 220, 220), 2)
        cv2.line(frame, (80, 0), (80, height), (255, 255, 255), 3)
        cv2.line(frame, (width - 80, 0), (width - 80, height), (255, 255, 255), 3)

        # Draw simulated counting line (dashed yellow)
        line_y = int(height * 0.60)
        cv2.line(frame, (80, line_y), (width - 80, line_y), (0, 255, 255), 2)

        # Update and render vehicles
        for obj in objects:
            current_y = int(obj["y"] + obj["speed"] * frame_idx)
            # Wrap around if moves past boundary
            if obj["speed"] > 0 and current_y > height + 50:
                current_y = (current_y % height) - 50
            elif obj["speed"] < 0 and current_y < -100:
                current_y = height + (current_y % height)

            x, w, h = obj["x"], obj["w"], obj["h"]
            cv2.rectangle(frame, (x, current_y), (x + w, current_y + h), obj["color"], -1)
            cv2.rectangle(frame, (x, current_y), (x + w, current_y + h), (255, 255, 255), 2)
            cv2.putText(
                frame,
                obj["label"],
                (x, max(current_y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Frame timestamp
        cv2.putText(
            frame,
            f"FRAME: {frame_idx + 1}/{total_frames} | SYNTHETIC TEST FEED",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    logger.info("Generated synthetic traffic video at: %s", output_path)
    return output_path


def get_best_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> Tuple[Optional[cv2.VideoWriter], Path]:
    """
    Attempt to initialize cv2.VideoWriter with preferred mp4v codec,
    falling back to XVID/avi if mp4v fails.

    Args:
        output_path: Target video file path.
        fps: Frames per second.
        width: Frame width.
        height: Frame height.

    Returns:
        (writer_instance_or_None, final_output_path)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    for fourcc_str, ext in fourcc_candidates:
        candidate_path = output_path.with_suffix(ext)
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer, candidate_path
            writer.release()
        except Exception as e:
            logger.debug("Codec %s failed: %s", fourcc_str, e)

    return None, output_path
