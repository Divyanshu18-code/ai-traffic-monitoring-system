"""
Video Input and Processing Module for AI Traffic Monitoring System.

Handles video ingestion from uploaded files, local webcams, and authorized IP/CCTV
streams with incremental frame extraction, metadata probing, safe output encoding,
and resource lifecycle management.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from config.settings import PROCESSED_DIR, get_config
from core.detector import Detection, YOLODetector
from core.tracker import ByteTracker
from core.vehicle_counter import VehicleCounter
from utils.file_utils import generate_unique_filepath
from utils.logger import sanitize_url, setup_logger
from utils.validation import (
    validate_file_size,
    validate_stream_url,
    validate_video_file_extension,
    validate_video_integrity,
)
from utils.video_utils import get_best_video_writer

logger = setup_logger("video_processor")


@dataclass
class ProcessingProgress:
    """Structured progress update emitted during video processing."""

    frame_index: int
    total_frames: int
    progress_percent: float
    elapsed_time: float
    current_fps: float
    detections_count: int
    is_completed: bool = False
    output_path: Optional[str] = None


@dataclass
class ProcessingResult:
    """Structured summary of a completed video processing run."""

    success: bool
    total_frames: int
    processed_frames: int
    elapsed_time_sec: float
    avg_fps: float
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    counting_summary: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class VideoProcessor:
    """
    Manages video ingestion, incremental frame reading, and video writing
    for files, webcams, and network streams.
    """

    def __init__(
        self,
        source: Union[str, int, Path],
        source_type: str = "video_file",
        output_filename: Optional[str] = None,
        save_output: bool = False,
    ) -> None:
        """
        Initialize VideoProcessor.

        Args:
            source: Path to video file, webcam integer index (e.g. 0), or RTSP/HTTP URL.
            source_type: 'video_file', 'webcam', or 'ip_camera'.
            output_filename: Optional name for saving processed video.
            save_output: Whether to encode and save annotated output video.
        """
        self.config = get_config()
        self.source = source
        self.source_type = source_type
        self.save_output = save_output
        self.output_filename = output_filename

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None

        self.width: int = 0
        self.height: int = 0
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.duration_sec: float = 0.0
        self.is_opened: bool = False

    def open(self) -> Tuple[bool, str]:
        """
        Open the video source and extract stream metadata.

        Returns:
            (success, status_or_error_message)
        """
        self.close()  # Ensure any previous session is released

        if self.source_type == "video_file":
            path = Path(str(self.source))
            if not path.exists():
                return False, f"Video file not found: {path}"

            ext_ok, ext_msg = validate_video_file_extension(path.name)
            if not ext_ok:
                return False, ext_msg

            size_ok, size_msg = validate_file_size(path.stat().st_size)
            if not size_ok:
                return False, size_msg

            integrity_ok, integrity_msg, meta = validate_video_integrity(path)
            if not integrity_ok:
                return False, integrity_msg

            self.cap = cv2.VideoCapture(str(path))
            self.width = meta.get("width", 640)
            self.height = meta.get("height", 480)
            self.fps = meta.get("fps", 30.0)
            self.total_frames = meta.get("total_frames", 0)
            self.duration_sec = meta.get("duration_sec", 0.0)

        elif self.source_type == "webcam":
            try:
                cam_index = int(self.source)
            except (ValueError, TypeError):
                cam_index = 0

            logger.info("Opening local webcam device: %d", cam_index)
            # Use DirectShow backend on Windows for faster initialization
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(cam_index)

            if not self.cap.isOpened():
                return False, f"Could not connect to webcam device index {cam_index}. Check permissions or device connection."

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
            if self.fps <= 0 or self.fps > 120:
                self.fps = 30.0
            self.total_frames = -1  # Live feed has indefinite frames
            self.duration_sec = 0.0

        elif self.source_type == "ip_camera":
            stream_url = str(self.source).strip().replace(" ", "")
            url_ok, url_msg = validate_stream_url(stream_url)
            if not url_ok:
                return False, url_msg

            sanitized = sanitize_url(stream_url)
            logger.info("Connecting to IP camera stream: %s", sanitized)

            # Auto-detect common IP Webcam endpoints (e.g. http://ip:8080 -> /video)
            candidates = [stream_url]
            try:
                from urllib.parse import urlparse
                parsed_u = urlparse(stream_url)
                if parsed_u.scheme in ("http", "https") and (not parsed_u.path or parsed_u.path in ("/", "")):
                    candidates.insert(0, stream_url.rstrip("/") + "/video")
            except Exception:
                pass

            opened = False
            for cand in candidates:
                self.cap = cv2.VideoCapture(cand)
                if self.cap.isOpened():
                    opened = True
                    break

            if not opened or self.cap is None or not self.cap.isOpened():
                return False, f"Failed to connect to IP camera stream at {sanitized}. Check that your phone and computer are on the same Wi-Fi network and verify the stream endpoint (e.g. {sanitized}/video)."

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
            if self.fps <= 0 or self.fps > 120:
                self.fps = 25.0
            self.total_frames = -1
            self.duration_sec = 0.0
        else:
            return False, f"Unsupported source type: {self.source_type}"

        if not self.cap.isOpened():
            return False, "Failed to initialize OpenCV VideoCapture."

        self.is_opened = True

        # Initialize output writer if saving requested
        if self.save_output:
            out_name = self.output_filename or f"output_{Path(str(self.source)).stem}.mp4"
            target_path = generate_unique_filepath(PROCESSED_DIR, out_name, prefix="proc")
            self.writer, self.output_path = get_best_video_writer(
                target_path, self.fps, self.width, self.height
            )
            if self.writer is None:
                logger.warning("Could not initialize video writer. Output will not be saved.")
            else:
                logger.info("Initialized output video writer at: %s", self.output_path)

        return True, "Video source opened successfully."

    def get_metadata(self) -> Dict[str, Any]:
        """Return video properties and stream metadata."""
        return {
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "is_live": self.total_frames <= 0,
            "output_path": str(self.output_path) if self.output_path else None,
        }

    def frame_generator(
        self,
        frame_skip: int = 0,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield frames incrementally from the video source.

        Args:
            frame_skip: Number of frames to skip between yields (0 = process every frame).
            stop_check: Optional callable that returns True when processing should halt.

        Yields:
            (frame_index, bgr_frame_numpy_array)
        """
        if not self.is_opened or self.cap is None:
            logger.error("Cannot read frames: video source is not open.")
            return

        frame_idx = 0
        consecutive_failures = 0
        max_failures = 10 if self.source_type != "video_file" else 3

        while True:
            if stop_check and stop_check():
                logger.info("Frame generation stopped by user request.")
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.info("End of stream reached or maximum read failures reached.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Frame skip logic
            if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
                frame_idx += 1
                continue

            yield frame_idx, frame
            frame_idx += 1

    def write_processed_frame(self, frame: np.ndarray) -> bool:
        """
        Write an annotated frame to the output video file.

        Args:
            frame: Annotated BGR frame.

        Returns:
            True if written, False otherwise.
        """
        if self.writer is not None and self.writer.isOpened():
            try:
                # Resize if frame dimensions do not match writer format
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self.writer.write(frame)
                return True
            except Exception as e:
                logger.error("Error writing frame to output video: %s", e)
                return False
        return False

    def process_frame(
        self,
        frame: np.ndarray,
        detector: YOLODetector,
        tracker: Optional[ByteTracker] = None,
        counter: Optional[VehicleCounter] = None,
        resize_dim: Optional[Tuple[int, int]] = None,
        show_track_id: bool = True,
        show_confidence: bool = True,
        show_counting_overlay: bool = True,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Process a single video frame with YOLO detection/tracking and counting overlay rendering.

        Args:
            frame: Input BGR numpy array.
            detector: Initialized YOLODetector instance.
            tracker: Optional ByteTracker instance for multi-object tracking.
            counter: Optional VehicleCounter instance for line-crossing counting.
            resize_dim: Optional (width, height) to resize frame before detection.
            show_track_id: Whether to display tracking ID.
            show_confidence: Whether to display confidence score on label.
            show_counting_overlay: Whether to render line and count HUD.

        Returns:
            Tuple of (annotated_frame, list_of_detections).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame, []

        processed = frame
        if resize_dim is not None and len(resize_dim) == 2:
            target_w, target_h = resize_dim
            if target_w > 0 and target_h > 0 and (frame.shape[1] != target_w or frame.shape[0] != target_h):
                processed = cv2.resize(frame, (target_w, target_h))

        # 1. Detection or Tracking
        if tracker is not None:
            detections = tracker.track(processed)
        else:
            detections = detector.detect(processed)

        # 2. Line Crossing Counting
        if counter is not None:
            counter.process_frame(detections, frame_shape=processed.shape[:2])

        # 3. Visualization Overlays
        if tracker is not None:
            annotated = tracker.draw_tracks(
                processed,
                detections,
                show_trails=True,
                show_confidence=show_confidence,
            )
        else:
            annotated = detector.draw_detections(
                processed,
                detections,
                show_track_id=show_track_id,
                show_confidence=show_confidence,
            )

        if counter is not None and show_counting_overlay:
            annotated = counter.draw_counting_overlay(annotated, show_line=True, show_hud=True)

        return annotated, detections

    def process_video(
        self,
        detector: YOLODetector,
        tracker: Optional[ByteTracker] = None,
        counter: Optional[VehicleCounter] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        frame_skip: int = 0,
        resize_dim: Optional[Tuple[int, int]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        db: Optional[Any] = None,
        session_id: Optional[str] = None,
        snapshot_interval: int = 30,
    ) -> ProcessingResult:
        """
        Run the end-to-end video processing pipeline with detection, tracking, and counting.

        Args:
            detector: YOLODetector instance.
            tracker: Optional ByteTracker instance.
            counter: Optional VehicleCounter instance.
            progress_callback: Optional callback receiving ProcessingProgress on each frame.
            frame_skip: Process every (frame_skip + 1)-th frame.
            resize_dim: Optional (width, height) resizing target.
            stop_check: Optional callable that halts processing early if True.
            db: Optional DatabaseManager instance for session persistence.
            session_id: Optional unique session ID. If not provided and db is active, one is generated.
            snapshot_interval: Frame interval for recording periodic traffic snapshots in db.

        Returns:
            ProcessingResult containing summary metrics, counting summary, output file path, and session_id.
        """
        if not self.is_opened:
            ok, msg = self.open()
            if not ok:
                return ProcessingResult(
                    success=False,
                    total_frames=0,
                    processed_frames=0,
                    elapsed_time_sec=0.0,
                    avg_fps=0.0,
                    output_path=None,
                    error_message=msg,
                )

        # Reset tracker and counter state for session isolation
        if tracker is not None:
            tracker.reset()
        if counter is not None:
            counter.reset()

        active_session_id = session_id
        if db is not None:
            if not active_session_id:
                active_session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            try:
                db.create_session(
                    session_id=active_session_id,
                    source_type=self.source_type,
                    source_name=Path(str(self.source)).name,
                    status="started",
                )
            except Exception as dbe:
                logger.warning("Database session start note: %s", dbe)

        start_time = time.time()
        processed_count = 0
        total_frames = self.total_frames

        try:
            for frame_idx, frame in self.frame_generator(frame_skip=frame_skip, stop_check=stop_check):
                frame_start = time.time()
                annotated, detections = self.process_frame(
                    frame=frame,
                    detector=detector,
                    tracker=tracker,
                    counter=counter,
                    resize_dim=resize_dim,
                )
                processed_count += 1

                if self.save_output:
                    self.write_processed_frame(annotated)

                now = time.time()
                elapsed = max(0.001, now - start_time)
                instant_fps = 1.0 / max(0.001, (now - frame_start))

                # Periodic snapshot recording
                if db is not None and active_session_id and (processed_count % snapshot_interval == 0):
                    try:
                        act_veh = sum(1 for d in detections if d.class_name in ("car", "bus", "truck", "motorcycle"))
                        c_in = counter.state.total_inbound if counter else 0
                        c_out = counter.state.total_outbound if counter else 0
                        c_tot = counter.state.total_crossed if counter else 0
                        cls_counts = counter.state.class_counts if counter else {}
                        db.insert_traffic_snapshot(
                            session_id=active_session_id,
                            frame_number=frame_idx,
                            active_vehicles=act_veh,
                            cumulative_inbound=c_in,
                            cumulative_outbound=c_out,
                            total_crossings=c_tot,
                            current_fps=instant_fps,
                            cars=cls_counts.get("car", 0),
                            buses=cls_counts.get("bus", 0),
                            trucks=cls_counts.get("truck", 0),
                            motorcycles=cls_counts.get("motorcycle", 0),
                        )
                    except Exception as dbe:
                        logger.debug("Database snapshot note: %s", dbe)

                progress_pct = (
                    round((processed_count / total_frames) * 100, 1)
                    if total_frames > 0
                    else 0.0
                )

                if progress_callback:
                    progress_callback(
                        ProcessingProgress(
                            frame_index=frame_idx,
                            total_frames=total_frames,
                            progress_percent=min(100.0, progress_pct),
                            elapsed_time=round(elapsed, 2),
                            current_fps=round(instant_fps, 1),
                            detections_count=len(detections),
                            output_path=str(self.output_path) if self.output_path else None,
                        )
                    )

            elapsed_total = max(0.001, time.time() - start_time)
            avg_fps = processed_count / elapsed_total

            final_out_str = None
            if self.save_output and self.output_path:
                self.close()  # Flush writer
                if self.output_path.exists() and self.output_path.stat().st_size > 0:
                    final_out_str = str(self.output_path)
                    logger.info("Processed video saved (%d bytes): %s", self.output_path.stat().st_size, final_out_str)
                else:
                    logger.warning("Output video file was not generated or is empty.")

            # Finalize database session
            if db is not None and active_session_id:
                try:
                    c_counts = counter.state.class_counts if counter else {}
                    db.finalize_session(
                        session_id=active_session_id,
                        duration_sec=elapsed_total,
                        total_vehicles=counter.state.total_unique_vehicles if counter else 0,
                        cars=c_counts.get("car", 0),
                        buses=c_counts.get("bus", 0),
                        trucks=c_counts.get("truck", 0),
                        motorcycles=c_counts.get("motorcycle", 0),
                        pedestrians=counter.state.total_unique_pedestrians if counter else 0,
                        total_crossings=counter.state.total_crossed if counter else 0,
                        inbound_count=counter.state.total_inbound if counter else 0,
                        outbound_count=counter.state.total_outbound if counter else 0,
                        avg_fps=avg_fps,
                        output_video_path=final_out_str,
                        status="completed",
                    )
                except Exception as dbe:
                    logger.warning("Database session finalization note: %s", dbe)

            return ProcessingResult(
                success=True,
                total_frames=total_frames,
                processed_frames=processed_count,
                elapsed_time_sec=round(elapsed_total, 2),
                avg_fps=round(avg_fps, 2),
                output_path=final_out_str,
                counting_summary=counter.get_summary() if counter is not None else None,
                session_id=active_session_id,
            )

        except Exception as e:
            logger.error("Error during video processing: %s", e)
            if db is not None and active_session_id:
                try:
                    db.update_session_status(active_session_id, status="failed", error_summary=str(e))
                except Exception:
                    pass

            return ProcessingResult(
                success=False,
                total_frames=total_frames,
                processed_frames=processed_count,
                elapsed_time_sec=round(time.time() - start_time, 2),
                avg_fps=round(processed_count / max(0.001, time.time() - start_time), 2),
                error_message=str(e),
                counting_summary=counter.get_summary() if counter is not None else None,
                session_id=active_session_id,
            )
        finally:
            self.close()

    def close(self) -> None:
        """Cleanly release OpenCV capture and writer resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture: %s", e)
            self.cap = None

        if self.writer is not None:
            try:
                self.writer.release()
            except Exception as e:
                logger.warning("Error releasing VideoWriter: %s", e)
            self.writer = None

        self.is_opened = False
        logger.debug("VideoProcessor resources released.")

    def __enter__(self) -> VideoProcessor:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

