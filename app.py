"""
AI-Powered Real-Time Traffic Monitoring and Vehicle Analytics System
Main Streamlit Application Dashboard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure PyTorch lib directory is explicitly in DLL directory search path on Windows
if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

# Import torch first on Windows before any UI or data libraries
import torch

import io
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import streamlit as st

from components.charts import (
    plot_class_distribution,
    plot_directional_traffic,
    plot_fps_performance,
    plot_traffic_trend,
)
from components.metrics import (
    render_congestion_badge,
    render_metric_card,
    render_summary_kpis,
)
from components.sidebar import render_sidebar
from components.video_panel import render_video_preview
from config.settings import DATA_DIR, SAMPLES_DIR, UPLOADS_DIR, get_config
from core.detector import YOLODetector
from core.tracker import ByteTracker
from core.traffic_analyzer import TrafficAnalyzer
from core.vehicle_counter import VehicleCounter
from core.video_processor import VideoProcessor
from database.db_manager import DatabaseManager
from utils.file_utils import generate_unique_filepath, sanitize_filename
from utils.logger import sanitize_url, setup_logger
from utils.validation import (
    validate_confidence_threshold,
    validate_file_size,
    validate_line_ratio,
    validate_stream_url,
    validate_video_file_extension,
    validate_video_integrity,
)
from utils.video_utils import generate_synthetic_traffic_video

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Traffic Monitoring & Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logger("streamlit_app")
config = get_config()


# Initialize Global Singletons & Resources
@st.cache_resource
def get_database_manager() -> DatabaseManager:
    """Retrieve or initialize singleton SQLite database manager."""
    return DatabaseManager()


@st.cache_resource
def load_yolo_detector(model_name: str, confidence: float, device: str) -> YOLODetector:
    """Cache loaded YOLO model weights in memory to avoid repeated loads."""
    return YOLODetector(model_name_or_path=model_name, confidence=confidence, device=device)


# Import custom components
from components.metrics import (
    render_congestion_badge,
    render_metric_card,
    render_sessions_html_table,
    render_summary_kpis,
)

# Custom CSS styling for modern dark charcoal aesthetic
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Background */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Top Header Bar Removal */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #101522 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Sidebar Radio Navigation as Modern Tiles */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 11px 16px !important;
        cursor: pointer !important;
        user-select: none !important;
        transition: all 0.15s ease !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        width: 100% !important;
        position: relative !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: rgba(0, 210, 106, 0.09) !important;
        border-color: rgba(0, 210, 106, 0.35) !important;
        color: #F8FAFC !important;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(0, 210, 106, 0.22), rgba(0, 210, 106, 0.04)) !important;
        border-left: 3px solid #00D26A !important;
        border-color: rgba(0, 210, 106, 0.5) !important;
        color: #00D26A !important;
    }
    /* Visually conceal radio input circle while preserving instant 1-click events */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    /* Inputs, Selects, and Date Picker */
    input[type="text"], input[type="number"], div[data-baseweb="input"] input, div[data-baseweb="select"] > div {
        background-color: #151C28 !important;
        border: 1px solid #2A3649 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    input[type="text"]:focus, input[type="number"]:focus, div[data-baseweb="select"] > div:focus-within {
        border-color: #00D26A !important;
        box-shadow: 0 0 0 2px rgba(0, 210, 106, 0.2) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="calendar"] {
        background-color: #151C28 !important;
        color: #F8FAFC !important;
    }

    /* Glassmorphism panels */
    div[data-testid="stExpander"], div[data-testid="stForm"] {
        background-color: #141A26 !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 10px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
    }

    /* Primary buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00D26A, #00A854) !important;
        color: #0B0F19 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00F07A, #00D26A) !important;
        box-shadow: 0 4px 18px rgba(0, 210, 106, 0.4) !important;
        color: #000 !important;
        transform: translateY(-1px);
    }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #00D26A !important;
        border-color: #00D26A !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00D26A !important;
        border-bottom-color: #00D26A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    db = get_database_manager()
    selected_page, conf_val, line_ratio_val = render_sidebar()

    # Model resolution
    model_name = st.session_state.get("selected_model_name", config.default_model_name)
    detector = load_yolo_detector(model_name, conf_val, config.device)

    # PAGE 1: OVERVIEW
    if selected_page == "Overview":
        render_overview_page(db)

    # PAGE 2: LIVE MONITORING
    elif selected_page == "Live Monitoring":
        render_live_monitoring_page(detector, db, conf_val, line_ratio_val)

    # PAGE 3: VIDEO ANALYSIS
    elif selected_page == "Video Analysis":
        render_video_analysis_page(detector, db, conf_val, line_ratio_val)

    # PAGE 4: TRAFFIC ANALYTICS
    elif selected_page == "Traffic Analytics":
        render_analytics_page(db)

    # PAGE 5: HISTORY
    elif selected_page == "History":
        render_history_page(db)

    # PAGE 6: SETTINGS
    elif selected_page == "Settings":
        render_settings_page(db)


def render_overview_page(db: DatabaseManager):
    """Render the dashboard overview with aggregate metrics and recent sessions."""
    st.markdown("## 🚦 Traffic Monitoring System Overview")
    st.markdown(
        "<p style='color: #94A3B8; margin-bottom: 24px;'>Real-time AI telemetry, object tracking, bidirectional counts, and vehicle flow audits.</p>",
        unsafe_allow_html=True,
    )

    aggregates = db.get_aggregate_metrics()
    recent_sessions = db.get_recent_sessions(limit=5)

    # Top KPI Metrics Row
    cols = st.columns(5)
    with cols[0]:
        render_metric_card("Total Sessions", aggregates.get("total_sessions", 0), "Processed audits", "#3B82F6", "📂")
    with cols[1]:
        render_metric_card("Total Vehicles", aggregates.get("total_vehicles", 0), "Unique motor vehicles", "#00D26A", "🚗")
    with cols[2]:
        render_metric_card("Inbound Count", aggregates.get("total_inbound", 0), "Entering sector", "#10B981", "⬇️")
    with cols[3]:
        render_metric_card("Outbound Count", aggregates.get("total_outbound", 0), "Exiting sector", "#F59E0B", "⬆️")
    with cols[4]:
        render_metric_card("System Throughput", f"{aggregates.get('overall_avg_fps', 0.0):.1f} FPS", "Mean inference rate", "#06B6D4", "⚡")

    st.markdown("---")

    # Visual Charts Row
    chart_cols = st.columns([1, 1])

    with chart_cols[0]:
        st.markdown("#### Vehicle Category Distribution")
        class_counts = {
            "car": aggregates.get("total_cars", 0),
            "bus": aggregates.get("total_buses", 0),
            "truck": aggregates.get("total_trucks", 0),
            "motorcycle": aggregates.get("total_motorcycles", 0),
            "person": aggregates.get("total_pedestrians", 0),
        }
        fig_classes = plot_class_distribution(class_counts)
        st.pyplot(fig_classes, use_container_width=True)

    with chart_cols[1]:
        st.markdown("#### Directional Balance")
        in_flow = aggregates.get("total_inbound", 0)
        out_flow = aggregates.get("total_outbound", 0)
        fig_flow = plot_directional_traffic(in_flow, out_flow)
        st.pyplot(fig_flow, use_container_width=True)

    st.markdown("---")

    # Recent Sessions Table
    st.markdown("#### Recent Processing Sessions")
    if recent_sessions:
        render_sessions_html_table(recent_sessions)
    else:
        st.info("No recorded processing sessions yet. Head to 'Video Analysis' or 'Live Monitoring' to run a session.")


def render_live_monitoring_page(
    detector: YOLODetector,
    db: DatabaseManager,
    conf_val: float,
    line_ratio: float,
):
    """Render the live webcam / IP stream monitoring interface."""
    st.markdown("## 📹 Live Stream Monitoring")
    st.markdown(
        "<p style='color: #94A3B8;'>Real-time computer vision inference from webcams and authorized RTSP/HTTP camera streams.</p>",
        unsafe_allow_html=True,
    )

    # Input Stream Configuration Panel
    with st.expander("Camera Stream Configuration", expanded=True):
        stream_cols = st.columns([1, 2, 1])
        with stream_cols[0]:
            source_type = st.selectbox(
                "Source Type",
                options=["webcam", "ip_camera"],
                format_func=lambda x: "Local Webcam" if x == "webcam" else "IP / CCTV Stream (RTSP/HTTP)",
            )

        with stream_cols[1]:
            if source_type == "webcam":
                cam_index = st.number_input("Webcam Device Index", min_value=0, max_value=4, value=0, step=1)
                source_input = int(cam_index)
                source_label = f"webcam_{cam_index}"
            else:
                rtsp_url = st.text_input(
                    "Stream URL",
                    placeholder="rtsp://username:password@192.168.1.50:554/live",
                    help="Enter authorized RTSP or HTTP video stream. Credentials will be sanitized.",
                )
                source_input = rtsp_url
                source_label = sanitize_url(rtsp_url)

        with stream_cols[2]:
            save_live_output = st.checkbox("Save Live Video", value=False, help="Record annotated output video to disk")

    # Control buttons
    btn_cols = st.columns([1, 1, 3])
    with btn_cols[0]:
        start_btn = st.button("▶️ Start Monitoring", key="start_live")
    with btn_cols[1]:
        stop_btn = st.button("⏹️ Stop Monitoring", key="stop_live")

    # State tracking
    if "is_monitoring" not in st.session_state:
        st.session_state["is_monitoring"] = False

    if start_btn:
        st.session_state["is_monitoring"] = True
    if stop_btn:
        st.session_state["is_monitoring"] = False

    # Layout for live preview and real-time statistics
    layout_cols = st.columns([3, 2])
    frame_placeholder = layout_cols[0].empty()
    stats_container = layout_cols[1].container()

    if not st.session_state["is_monitoring"]:
        frame_placeholder.markdown(
            """
            <div style="
                border: 2px dashed rgba(255, 255, 255, 0.15);
                border-radius: 10px;
                padding: 100px 20px;
                text-align: center;
                color: #64748B;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📷</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #94A3B8;">Live Stream Inactive</div>
                <div style="font-size: 0.85rem; margin-top: 6px;">Configure your camera above and click 'Start Monitoring' to begin.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Initialize Live Pipeline
    vp = VideoProcessor(
        source=source_input,
        source_type=source_type,
        output_filename=f"live_{uuid.uuid4().hex[:6]}.mp4",
        save_output=save_live_output,
    )
    ok, msg = vp.open()
    if not ok:
        st.error(f"Failed to start camera feed: {msg}")
        st.session_state["is_monitoring"] = False
        return

    tracker = ByteTracker(detector)
    counter = VehicleCounter(line_ratio=line_ratio)
    analyzer = TrafficAnalyzer()

    session_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    db.create_session(session_id, source_type, source_label, notes="Live monitoring session")

    st.success(f"Connected to stream. Monitoring Session: `{session_id}`")

    try:
        frame_count = 0
        for frame_idx, frame in vp.frame_generator(stop_check=lambda: not st.session_state.get("is_monitoring", False)):
            t_start = time.time()
            frame_count += 1

            # 1. ByteTrack Object Tracking
            tracks = tracker.track(frame)

            # 2. Line Crossing & Counting
            counting_state = counter.process_frame(tracks, frame_shape=frame.shape[:2])

            # 3. Analytics & Congestion
            t_elapsed = max(0.001, time.time() - t_start)
            snapshot = analyzer.record_frame(
                frame_idx, tracks, counting_state, frame.shape[:2], t_elapsed
            )

            # Periodic snapshot to SQLite (every 30 frames)
            if frame_count % 30 == 0:
                db.save_snapshot(
                    session_id=session_id,
                    frame_number=frame_idx,
                    active_vehicles=snapshot.active_vehicles,
                    congestion_level=snapshot.congestion_level,
                    cumulative_inbound=snapshot.cumulative_inbound,
                    cumulative_outbound=snapshot.cumulative_outbound,
                    current_fps=snapshot.current_fps,
                )

            # 4. Draw Overlays
            annotated = tracker.draw_tracks(frame, tracks, show_trails=True)
            annotated = counter.draw_counting_overlay(annotated, show_line=True, show_hud=True)

            # 5. Write to output video if recording enabled
            if save_live_output:
                vp.write_processed_frame(annotated)

            # 6. Render live frame
            render_video_preview(frame_placeholder, annotated)

            # 7. Render dynamic stats
            with stats_container:
                st.markdown(f"**Session**: `{session_id}`")
                render_congestion_badge(snapshot.congestion_level)

                st.markdown("---")
                stat_cols = st.columns(2)
                with stat_cols[0]:
                    render_metric_card("Inbound", counting_state.total_inbound, "Entering sector", "#3B82F6", "⬇️")
                    render_metric_card("Active Cars", counting_state.class_counts.get("car", 0), "Currently tracked", "#00D26A", "🚗")
                with stat_cols[1]:
                    render_metric_card("Outbound", counting_state.total_outbound, "Exiting sector", "#F59E0B", "⬆️")
                    render_metric_card("Live FPS", f"{snapshot.current_fps:.1f}", "Processing rate", "#06B6D4", "⚡")

    finally:
        vp.close()
        summary = analyzer.get_analytics_summary(counter.state)
        db.finalize_session(
            session_id=session_id,
            duration_sec=summary["session_duration_sec"],
            total_vehicles=summary["total_unique_vehicles"],
            cars=counter.state.class_counts.get("car", 0),
            buses=counter.state.class_counts.get("bus", 0),
            trucks=counter.state.class_counts.get("truck", 0),
            motorcycles=counter.state.class_counts.get("motorcycle", 0),
            pedestrians=summary["total_unique_pedestrians"],
            inbound_count=summary["inbound_count"],
            outbound_count=summary["outbound_count"],
            avg_fps=summary["average_fps"],
            output_video_path=str(vp.output_path) if vp.output_path else None,
        )
        st.info("Live monitoring stopped. Session statistics recorded in database.")


def render_video_analysis_page(
    detector: YOLODetector,
    db: DatabaseManager,
    conf_val: float,
    line_ratio: float,
):
    """Render the video file upload, analysis, and download interface."""
    st.markdown("## 🎬 Video File Analysis")
    st.markdown(
        "<p style='color: #94A3B8;'>Upload pre-recorded traffic video or use synthetic sample test feeds for comprehensive auditing.</p>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_sample = st.tabs(["📁 Upload Traffic Video", "🧪 Use Synthetic Sample Video"])

    selected_video_path: Optional[Path] = None
    original_name: str = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Select video file",
            type=["mp4", "avi", "mov", "mkv"],
            help=f"Supported formats: mp4, avi, mov, mkv. Max size: {config.max_upload_size_mb} MB",
        )
        if uploaded_file is not None:
            original_name = uploaded_file.name
            safe_name = sanitize_filename(original_name)
            target_path = UPLOADS_DIR / safe_name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_video_path = target_path

    with tab_sample:
        st.markdown(
            "Select a preloaded traffic video feed with real vehicles for accurate YOLO object detection and counting."
        )
        sample_options = {}
        real_sample = SAMPLES_DIR / "sample_traffic_feed.mp4"
        if real_sample.exists():
            sample_options["🚗 Real Traffic Highway Feed (Recommended - Real Cars & Crossings)"] = real_sample

        multi_sample = SAMPLES_DIR / "multi_traffic_sample.mp4"
        if multi_sample.exists():
            sample_options["🚶 Mixed Traffic Feed (Cars & Pedestrians)"] = multi_sample

        if sample_options:
            chosen_label = st.radio(
                "Select Traffic Sample Feed:",
                list(sample_options.keys()),
                index=0,
                key="sample_feed_choice",
            )
            selected_video_path = sample_options[chosen_label]
            original_name = selected_video_path.name
            st.success(f"Selected: **{chosen_label}** (`{original_name}`)")
        else:
            st.warning("No preloaded sample videos found in `data/samples/`.")

    if selected_video_path is None:
        st.info("Please upload a video file or select a sample feed above to proceed.")
        return

    # Probe and validate selected video
    valid_ok, valid_msg, meta = validate_video_integrity(selected_video_path)
    if not valid_ok:
        st.error(f"Video validation failed: {valid_msg}")
        return

    # Video Info Cards
    st.markdown("#### Input Video Properties")
    meta_cols = st.columns(4)
    with meta_cols[0]:
        render_metric_card("Dimensions", f"{meta['width']} x {meta['height']}", "Pixel resolution", "#3B82F6", "📐")
    with meta_cols[1]:
        render_metric_card("FPS", f"{meta['fps']}", "Recorded frame rate", "#10B981", "⏱️")
    with meta_cols[2]:
        render_metric_card("Total Frames", meta["total_frames"], "Frame count", "#8B5CF6", "🎞️")
    with meta_cols[3]:
        render_metric_card("Duration", f"{meta['duration_sec']}s", "Video runtime", "#F59E0B", "⏳")

    # Processing options
    st.markdown("#### Analysis Configuration")
    cfg_cols = st.columns(3)
    with cfg_cols[0]:
        save_annotated = st.checkbox("Encode & Save Annotated Output Video", value=True)
    with cfg_cols[1]:
        show_live_preview = st.checkbox("Show Preview During Processing", value=True)
    with cfg_cols[2]:
        frame_skip = st.number_input("Process Every Nth Frame", min_value=1, max_value=5, value=1) - 1

    if st.button("🚀 Start Video Analysis", type="primary"):
        process_video_file(
            video_path=selected_video_path,
            original_filename=original_name,
            detector=detector,
            db=db,
            line_ratio=line_ratio,
            save_annotated=save_annotated,
            show_preview=show_live_preview,
            frame_skip=int(frame_skip),
        )


def process_video_file(
    video_path: Path,
    original_filename: str,
    detector: YOLODetector,
    db: DatabaseManager,
    line_ratio: float,
    save_annotated: bool,
    show_preview: bool,
    frame_skip: int,
):
    """Execute complete object detection, tracking, and counting pipeline on a video file."""
    vp = VideoProcessor(
        source=video_path,
        source_type="video_file",
        output_filename=f"annotated_{video_path.stem}.mp4",
        save_output=save_annotated,
    )
    ok, msg = vp.open()
    if not ok:
        st.error(f"Failed to open video: {msg}")
        return

    tracker = ByteTracker(detector)
    counter = VehicleCounter(line_ratio=line_ratio)
    analyzer = TrafficAnalyzer()

    session_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    db.create_session(session_id, "video_file", original_filename, notes="Automated video analysis")

    progress_bar = st.progress(0, text="Initializing video analysis...")
    preview_placeholder = st.empty() if show_preview else None
    status_placeholder = st.empty()

    total_frames = max(1, vp.total_frames)
    processed_count = 0

    try:
        for frame_idx, frame in vp.frame_generator(frame_skip=frame_skip):
            t0 = time.time()
            processed_count += 1

            # 1. ByteTrack Object Tracking
            tracks = tracker.track(frame)

            # 2. Line Crossing & Counting
            counting_state = counter.process_frame(tracks, frame_shape=frame.shape[:2])

            # 3. Analytics & Performance
            t_elapsed = max(0.001, time.time() - t0)
            snapshot = analyzer.record_frame(
                frame_idx, tracks, counting_state, frame.shape[:2], t_elapsed
            )

            # Save snapshot every 20 frames
            if processed_count % 20 == 0:
                db.save_snapshot(
                    session_id=session_id,
                    frame_number=frame_idx,
                    active_vehicles=snapshot.active_vehicles,
                    congestion_level=snapshot.congestion_level,
                    cumulative_inbound=snapshot.cumulative_inbound,
                    cumulative_outbound=snapshot.cumulative_outbound,
                    current_fps=snapshot.current_fps,
                )

            # 4. Overlay rendering
            annotated = tracker.draw_tracks(frame, tracks, show_trails=True)
            annotated = counter.draw_counting_overlay(annotated, show_line=True, show_hud=True)

            if save_annotated:
                vp.write_processed_frame(annotated)

            # Update progress bar
            pct = min(1.0, frame_idx / total_frames)
            progress_bar.progress(
                pct,
                text=f"Processing Frame {frame_idx}/{total_frames} ({int(pct * 100)}%) | {snapshot.current_fps:.1f} FPS",
            )

            if show_preview and preview_placeholder and (processed_count % 2 == 0):
                render_video_preview(preview_placeholder, annotated)

        progress_bar.progress(1.0, text="Processing Complete!")
        status_placeholder.success("🎉 Video analysis successfully completed!")

    finally:
        vp.close()
        summary = analyzer.get_analytics_summary(counter.state)
        db.finalize_session(
            session_id=session_id,
            duration_sec=summary["session_duration_sec"],
            total_vehicles=summary["total_unique_vehicles"],
            cars=counter.state.class_counts.get("car", 0),
            buses=counter.state.class_counts.get("bus", 0),
            trucks=counter.state.class_counts.get("truck", 0),
            motorcycles=counter.state.class_counts.get("motorcycle", 0),
            pedestrians=summary["total_unique_pedestrians"],
            inbound_count=summary["inbound_count"],
            outbound_count=summary["outbound_count"],
            avg_fps=summary["average_fps"],
            output_video_path=str(vp.output_path) if vp.output_path else None,
        )

    # Display final session results
    st.markdown("### 📊 Final Session Analytics")
    render_summary_kpis(summary)

    res_cols = st.columns(2)
    with res_cols[0]:
        st.pyplot(plot_class_distribution(counter.state.class_counts), use_container_width=True)
    with res_cols[1]:
        st.pyplot(plot_directional_traffic(summary["inbound_count"], summary["outbound_count"]), use_container_width=True)

    # Downloads & Exports (Phase 9)
    st.markdown("### 📥 Export Reports & Video")
    dl_cols = st.columns(2)

    with dl_cols[0]:
        # Export Session Summary CSV
        session_data = {
            "session_id": [session_id],
            "original_filename": [original_filename],
            "duration_sec": [summary["session_duration_sec"]],
            "total_unique_vehicles": [summary["total_unique_vehicles"]],
            "cars": [counter.state.class_counts.get("car", 0)],
            "buses": [counter.state.class_counts.get("bus", 0)],
            "trucks": [counter.state.class_counts.get("truck", 0)],
            "motorcycles": [counter.state.class_counts.get("motorcycle", 0)],
            "pedestrians": [summary["total_unique_pedestrians"]],
            "inbound_count": [summary["inbound_count"]],
            "outbound_count": [summary["outbound_count"]],
            "average_fps": [summary["average_fps"]],
        }
        df_sess = pd.DataFrame(session_data)
        csv_buffer = io.StringIO()
        df_sess.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download Session Metrics (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"session_{session_id}.csv",
            mime="text/csv",
        )

    with dl_cols[1]:
        # Download Processed Video if exists
        if vp.output_path and vp.output_path.exists():
            with open(vp.output_path, "rb") as vf:
                video_bytes = vf.read()
            st.download_button(
                label="🎬 Download Annotated Video",
                data=video_bytes,
                file_name=vp.output_path.name,
                mime="video/mp4",
            )
        else:
            st.info("No output video file generated.")


def render_analytics_page(db: DatabaseManager):
    """Render comprehensive traffic analytics and trend charts."""
    st.markdown("## 📈 Deep Traffic Analytics")
    st.markdown(
        "<p style='color: #94A3B8;'>Temporal trends, density audits, and historical congestion patterns.</p>",
        unsafe_allow_html=True,
    )

    all_sessions = db.get_all_sessions()
    if not all_sessions:
        st.warning("No historical sessions available for analytics. Process a video or run live monitoring first.")
        return

    # Session Selector
    session_options = {s["session_id"]: f"{s['session_id']} ({s['source_name']} - {s['start_time']})" for s in all_sessions}
    selected_sess_id = st.selectbox(
        "Select Session to Analyze",
        options=list(session_options.keys()),
        format_func=lambda sid: session_options[sid],
    )

    session = db.get_session(selected_sess_id)
    snapshots = db.get_snapshots_for_session(selected_sess_id)

    if session:
        st.markdown(f"### Session Details: `{selected_sess_id}`")
        render_summary_kpis(session)

        st.markdown("---")
        chart_cols = st.columns(2)

        with chart_cols[0]:
            class_counts = {
                "car": session.get("cars", 0),
                "bus": session.get("buses", 0),
                "truck": session.get("trucks", 0),
                "motorcycle": session.get("motorcycles", 0),
                "person": session.get("pedestrians", 0),
            }
            st.pyplot(plot_class_distribution(class_counts), use_container_width=True)

        with chart_cols[1]:
            st.pyplot(
                plot_directional_traffic(session.get("inbound_count", 0), session.get("outbound_count", 0)),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Temporal Snapshots & Throughput")
        if snapshots:
            trend_fig = plot_traffic_trend(snapshots)
            if trend_fig:
                st.pyplot(trend_fig, use_container_width=True)

            fps_fig = plot_fps_performance(snapshots)
            if fps_fig:
                st.pyplot(fps_fig, use_container_width=True)
        else:
            st.info("No granular frame snapshots recorded for this session.")


def render_history_page(db: DatabaseManager):
    """Render historical session search, filters, and full CSV export."""
    st.markdown("## 📜 Processing History & Audits")
    st.markdown(
        "<p style='color: #94A3B8;'>Filter, search, inspect, and export all recorded traffic monitoring sessions.</p>",
        unsafe_allow_html=True,
    )

    filter_cols = st.columns(3)
    with filter_cols[0]:
        search_query = st.text_input("Search Filename / ID", placeholder="Filter by name...")
    with filter_cols[1]:
        src_filter = st.selectbox("Source Type", options=["All", "video_file", "webcam", "ip_camera"])
    with filter_cols[2]:
        date_filter = st.date_input("Filter Date", value=None)

    date_str = date_filter.strftime("%Y-%m-%d") if date_filter else None
    sessions = db.filter_sessions(start_date=date_str, end_date=date_str, source_type=src_filter)

    if search_query:
        query = search_query.lower()
        sessions = [
            s
            for s in sessions
            if query in s.get("session_id", "").lower() or query in s.get("source_name", "").lower()
        ]

    # Full Export Button (Phase 9)
    export_cols = st.columns([3, 1])
    with export_cols[1]:
        all_df = db.export_sessions_dataframe()
        if not all_df.empty:
            csv_data = all_df.to_csv(index=False)
            st.download_button(
                label="📥 Export All Records (CSV)",
                data=csv_data,
                file_name="traffic_monitoring_sessions_export.csv",
                mime="text/csv",
                type="primary",
            )

    if sessions:
        render_sessions_html_table(sessions)
        with st.expander("🔍 Interactive Data Table & Raw Columns", expanded=False):
            df = pd.DataFrame(sessions)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No sessions match the selected filter criteria.")


def render_settings_page(db: DatabaseManager):
    """Render system configuration, hardware status, and database diagnostics."""
    st.markdown("## ⚙️ System Settings & Diagnostics")
    st.markdown(
        "<p style='color: #94A3B8;'>Configure detection parameters, model weights, and inspect database state.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Object Detection Settings")
        selected_model = st.selectbox(
            "YOLO Model Weights",
            options=config.available_models,
            index=0,
            help="Select YOLO model architecture. Nano is recommended for CPU inference.",
        )
        st.session_state["selected_model_name"] = selected_model

        conf_slider = st.slider(
            "Default Confidence Threshold",
            0.10, 0.90, config.confidence_threshold, 0.05
        )
        iou_slider = st.slider("IoU NMS Threshold", 0.10, 0.90, config.iou_threshold, 0.05)

        st.markdown("### Congestion Classification Heuristics")
        low_t = st.number_input("Low Threshold (Active Vehicles)", min_value=1, max_value=20, value=config.congestion_low_threshold)
        mod_t = st.number_input("Moderate Threshold (Active Vehicles)", min_value=low_t + 1, max_value=50, value=config.congestion_moderate_threshold)

    with col2:
        st.markdown("### Hardware & Compute Diagnostics")
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None (CPU Execution)"
        st.info(f"**Execution Device:** `{config.device}`\n\n**CUDA Available:** `{cuda_ok}`\n\n**GPU Hardware:** `{gpu_name}`\n\n**PyTorch Version:** `{torch.__version__}`")

        st.markdown("### Database Diagnostics")
        db_exists = config.db_path.exists()
        db_size_mb = config.db_path.stat().st_size / (1024 * 1024) if db_exists else 0.0
        st.write(f"**Database File:** `{config.db_path}`")
        st.write(f"**Database Size:** `{db_size_mb:.2f} MB`")

        if st.button("Verify Database Integrity"):
            try:
                metrics = db.get_aggregate_metrics()
                st.success(f"Database operational. Total Recorded Sessions: {metrics.get('total_sessions', 0)}")
            except Exception as e:
                st.error(f"Database error: {e}")


if __name__ == "__main__":
    main()
