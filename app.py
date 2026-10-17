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
import importlib
import streamlit as st

import components.charts
import components.metrics
import components.sidebar
import components.video_panel

importlib.reload(components.charts)
importlib.reload(components.metrics)
importlib.reload(components.sidebar)
importlib.reload(components.video_panel)

from components.charts import (
    plot_class_distribution,
    plot_directional_traffic,
    plot_fps_performance,
    plot_sessions_history,
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
if "pytest" not in sys.modules:
    try:
        import core.tracker
        importlib.reload(core.tracker)
    except Exception:
        pass

from core.detector import YOLODetector
from core.tracker import ByteTracker
from core.traffic_analyzer import TrafficAnalyzer
from core.vehicle_counter import VehicleCounter
from core.video_processor import VideoProcessor
from database.db_manager import DatabaseManager
from utils.file_utils import format_bytes, generate_unique_filepath, sanitize_filename
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
def _create_database_manager() -> DatabaseManager:
    """Instantiate fresh database manager."""
    return DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """Retrieve singleton SQLite database manager, refreshing if stale."""
    global DatabaseManager
    mgr = _create_database_manager()
    # Check if instance is from a stale cache (missing recent methods)
    if not hasattr(mgr, "update_session_status"):
        try:
            import database.db_manager
            importlib.reload(database.db_manager)
            from database.db_manager import DatabaseManager as FreshDatabaseManager
            DatabaseManager = FreshDatabaseManager
        except Exception:
            pass
        st.cache_resource.clear()
        mgr = _create_database_manager()
    return mgr


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
# Custom CSS styling for modern dark charcoal aesthetic & responsive mobile phone view
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography & Smooth Background */
    html {
        scroll-behavior: smooth;
    }
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151D2C 0%, #080C14 75%) !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Modern Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #080C14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00D26A;
    }
    
    /* Top Header Bar Removal */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
        transition: padding 0.3s ease;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
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
        border-radius: 10px !important;
        padding: 11px 16px !important;
        cursor: pointer !important;
        user-select: none !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        width: 100% !important;
        position: relative !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: rgba(0, 210, 106, 0.09) !important;
        border-color: rgba(0, 210, 106, 0.4) !important;
        color: #F8FAFC !important;
        transform: translateX(4px);
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(0, 210, 106, 0.22), rgba(0, 210, 106, 0.04)) !important;
        border-left: 3.5px solid #00D26A !important;
        border-color: rgba(0, 210, 106, 0.5) !important;
        color: #00D26A !important;
        box-shadow: 0 4px 16px rgba(0, 210, 106, 0.12) !important;
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

    /* =========================================================================
       COMPREHENSIVE DROPDOWN & SELECTBOX STYLING (FIXES ALL STREAMLIT DROPDOWNS)
       ========================================================================= */
    div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
        margin-bottom: 6px !important;
        letter-spacing: 0.01em !important;
    }

    /* Dropdown Trigger Box */
    div[data-baseweb="select"] {
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #121824 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        min-height: 44px !important;
        padding: 2px 8px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(0, 210, 106, 0.5) !important;
        background-color: #161F2E !important;
        box-shadow: 0 4px 14px rgba(0, 210, 106, 0.15) !important;
        transform: translateY(-1px);
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #00D26A !important;
        box-shadow: 0 0 0 2px rgba(0, 210, 106, 0.25), 0 4px 16px rgba(0, 210, 106, 0.2) !important;
        background-color: #161F2E !important;
    }

    /* Selected Value Text */
    div[data-baseweb="select"] span, div[data-baseweb="select"] div[aria-selected="true"] {
        color: #F8FAFC !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        user-select: none !important;
    }

    /* Completely Remove Blinking Text Caret / Cursor Line from all Dropdowns */
    div[data-testid="stSelectbox"] input,
    div[data-testid="stMultiSelect"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] input[role="combobox"],
    input[aria-autocomplete="list"],
    div[data-baseweb="select"] input:focus,
    div[data-testid="stSelectbox"] input:focus {
        caret-color: transparent !important;
        cursor: pointer !important;
        user-select: none !important;
        -webkit-user-select: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stSelectbox"] *,
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }

    /* Dropdown Arrow Icon */
    div[data-baseweb="select"] svg {
        fill: #00D26A !important;
        stroke: #00D26A !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Dropdown Menu Popover Container */
    div[data-baseweb="popover"] {
        background-color: #111622 !important;
        border: 1px solid rgba(0, 210, 106, 0.35) !important;
        border-radius: 12px !important;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        overflow: hidden !important;
        margin-top: 4px !important;
        z-index: 999999 !important;
    }
    div[data-baseweb="popover"] ul, ul[role="listbox"] {
        background-color: #111622 !important;
        padding: 6px !important;
        border: none !important;
    }
    div[data-baseweb="popover"] li, li[role="option"] {
        background-color: transparent !important;
        color: #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 3px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background: rgba(0, 210, 106, 0.16) !important;
        color: #00F07A !important;
        font-weight: 600 !important;
        transform: translateX(4px);
    }

    /* Regular Text & Number Inputs */
    input[type="text"], input[type="number"], div[data-baseweb="input"] input {
        background-color: #121824 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
        min-height: 42px !important;
        padding: 8px 14px !important;
        transition: all 0.2s ease !important;
    }
    input[type="text"]:focus, input[type="number"]:focus {
        border-color: #00D26A !important;
        box-shadow: 0 0 0 2px rgba(0, 210, 106, 0.25) !important;
        background-color: #161F2E !important;
    }

    /* Modern Glassmorphism Panels & Forms */
    div[data-testid="stExpander"], div[data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(20, 26, 38, 0.75), rgba(14, 18, 28, 0.85)) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(0, 210, 106, 0.25) !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4), 0 0 15px rgba(0, 210, 106, 0.06) !important;
    }

    /* Sleek File Uploader */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(18, 24, 36, 0.8), rgba(12, 16, 26, 0.9)) !important;
        border: 2px dashed rgba(0, 210, 106, 0.3) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #00D26A !important;
        background: rgba(0, 210, 106, 0.04) !important;
        box-shadow: 0 8px 28px rgba(0, 210, 106, 0.12) !important;
    }

    /* Primary Glowing CTA Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00D26A, #00A854) !important;
        color: #070B12 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 16px rgba(0, 210, 106, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        letter-spacing: 0.01em !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00F07A, #00D26A) !important;
        box-shadow: 0 6px 24px rgba(0, 210, 106, 0.45) !important;
        color: #000 !important;
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: translateY(0px) scale(0.98);
    }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #00D26A !important;
        border-color: #00D26A !important;
        box-shadow: 0 0 10px rgba(0, 210, 106, 0.5) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00D26A !important;
        border-bottom-color: #00D26A !important;
    }

    /* Metric Cards Hover Lift */
    .custom-metric-card {
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease, border-color 0.25s ease !important;
    }
    .custom-metric-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45), 0 0 20px rgba(0, 210, 106, 0.12) !important;
    }

    /* =========================================================================
       RESPONSIVE PHONE / MOBILE VIEW (REAL WEBSITE MOBILE EXPERIENCE)
       ========================================================================= */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            max-width: 100% !important;
        }

        /* Stack side-by-side columns into clean full-width cards on phones */
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 12px !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 6px !important;
        }

        /* Full-width touch friendly buttons */
        .stButton>button {
            width: 100% !important;
            min-height: 48px !important;
            font-size: 1rem !important;
            margin-bottom: 8px !important;
            justify-content: center !important;
        }

        /* Touch friendly inputs and selects */
        input[type="text"], input[type="number"], div[data-baseweb="select"] > div {
            min-height: 48px !important;
            font-size: 0.95rem !important;
        }

        /* Mobile Headings Scaling */
        h1 { font-size: 1.55rem !important; line-height: 1.25 !important; }
        h2 { font-size: 1.3rem !important; line-height: 1.3 !important; }
        h3 { font-size: 1.15rem !important; }
        h4 { font-size: 1rem !important; }

        /* Full width responsive video and images */
        img, video, div[data-testid="stImage"] img {
            max-width: 100% !important;
            height: auto !important;
            border-radius: 10px !important;
        }

        /* Sidebar width on mobile */
        section[data-testid="stSidebar"] {
            width: 85vw !important;
            min-width: 280px !important;
        }
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
    """Render the dashboard overview with aggregate metrics, latest session highlight, and recent trends."""
    st.markdown("## 🚦 Traffic Monitoring System Overview")
    st.markdown(
        "<p style='color: #94A3B8; margin-bottom: 24px;'>Real-time AI telemetry, object tracking, bidirectional counts, and vehicle flow audits.</p>",
        unsafe_allow_html=True,
    )

    aggregates = db.get_aggregate_metrics()
    recent_sessions = db.get_recent_sessions(limit=8)
    total_sessions = aggregates.get("total_sessions", 0)

    if total_sessions == 0:
        st.info(
            "👋 **Welcome to AI Traffic Monitoring System!**\n\n"
            "No monitoring sessions have been recorded in the database yet. "
            "Head over to **📹 Video Analysis** to process a traffic video or **📹 Live Monitoring** to start a live feed."
        )
        return

    # Top KPI Metrics Row with accurate labeling
    cols = st.columns(5)
    with cols[0]:
        render_metric_card("Total Sessions", total_sessions, "Completed audit sessions", "#3B82F6", "📂")
    with cols[1]:
        tot_veh = aggregates.get("total_vehicles", 0)
        render_metric_card("Monitored Vehicles", tot_veh, "Sum of session observations", "#00D26A", "🚗")
    with cols[2]:
        tot_cross = aggregates.get("total_crossings", aggregates.get("total_inbound", 0) + aggregates.get("total_outbound", 0))
        render_metric_card("Line Crossings", tot_cross, "Total crossing events", "#8B5CF6", "⚡")
    with cols[3]:
        in_flow = aggregates.get("total_inbound", 0)
        out_flow = aggregates.get("total_outbound", 0)
        render_metric_card("Inbound / Outbound", f"{in_flow} / {out_flow}", "Sector flow balance", "#10B981", "↔️")
    with cols[4]:
        fps = aggregates.get("overall_avg_fps", 0.0)
        render_metric_card("System Throughput", f"{fps:.1f} FPS", "Mean inference rate", "#06B6D4", "🚀")

    # Latest Session Highlight Card
    latest_sess = recent_sessions[0] if recent_sessions else None
    if latest_sess:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-left: 4px solid #3B82F6;
                border-radius: 8px;
                padding: 14px 20px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 12px;
            ">
                <div>
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">LATEST COMPLETED SESSION</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                        {latest_sess.get('source_name', 'Unnamed Feed')}
                        <span style="font-size: 0.8rem; font-family: monospace; color: #38BDF8; margin-left: 8px;">({latest_sess.get('session_id', '—')})</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 2px;">
                        Recorded: {latest_sess.get('start_time', '—')} | Runtime: {latest_sess.get('duration_sec', 0.0):.1f}s | Status: <span style="color: #10B981; font-weight: 600;">{latest_sess.get('status', 'completed').upper()}</span>
                    </div>
                </div>
                <div style="display: flex; gap: 20px;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8;">VEHICLES</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #00D26A;">{latest_sess.get('total_vehicles', 0)}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8;">INBOUND</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #3B82F6;">{latest_sess.get('inbound_count', 0)}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8;">OUTBOUND</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #F59E0B;">{latest_sess.get('outbound_count', 0)}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8;">AVG FPS</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #06B6D4;">{latest_sess.get('avg_fps', 0.0):.1f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        fig_flow = plot_directional_traffic(in_flow, out_flow)
        st.pyplot(fig_flow, use_container_width=True)

    # Multi-session flow comparison chart if multiple sessions
    if len(recent_sessions) > 1:
        st.markdown("#### Historical Sessions Flow Comparison")
        fig_hist = plot_sessions_history(recent_sessions)
        if fig_hist:
            st.pyplot(fig_hist, use_container_width=True)

    st.markdown("---")

    # Recent Sessions Table
    st.markdown("#### Recent Processing Sessions")
    render_sessions_html_table(recent_sessions)


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
                    placeholder="e.g. http://10.176.215.18:8080/video or rtsp://...",
                    help="For mobile 'IP Webcam' app, use http://IP:PORT/video (e.g. http://10.176.215.18:8080/video)",
                )
                clean_url = rtsp_url.strip().replace(" ", "") if rtsp_url else ""
                source_input = clean_url
                source_label = sanitize_url(clean_url)

        with stream_cols[2]:
            save_live_output = st.checkbox("Save Live Video", value=False, help="Record annotated output video to disk")
            detect_all_objects = st.checkbox(
                "Detect Everything (All Classes)",
                value=True,
                help="Detect all 80 COCO classes (person, cell phone, laptop, bottle, chair, cup, etc.) in the live camera feed.",
            )
            max_frames_choice = st.selectbox(
                "Demo Frame Limit",
                options=["Continuous (Manual Stop)", "150 frames (~5-10s)", "300 frames (~15s)", "600 frames (~30s)"],
                index=0,
                help="Set a bounded frame limit suitable for quick demonstration or continuous live monitoring",
            )

    limit_map = {
        "150 frames (~5-10s)": 150,
        "300 frames (~15s)": 300,
        "600 frames (~30s)": 600,
    }
    max_frames_limit = limit_map.get(max_frames_choice, None)

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
    stats_placeholder = layout_cols[1].empty()

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
        st.error(f"⚠️ Unable to open camera feed: {msg}")
        st.info(
            "💡 **Hardware Note**: If this system does not have an active physical webcam or live CCTV stream, "
            "you can use the **🎬 Video Analysis** page to test real traffic feeds with detection, tracking, and vehicle counting."
        )
        st.session_state["is_monitoring"] = False
        return

    if "pytest" not in sys.modules:
        try:
            import core.tracker
            importlib.reload(core.tracker)
            from core.tracker import ByteTracker as LiveByteTracker
            tracker = LiveByteTracker(detector)
        except Exception:
            tracker = ByteTracker(detector)
    else:
        tracker = ByteTracker(detector)

    analyzer = TrafficAnalyzer()

    session_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    db.create_session(session_id, source_type, source_label, notes="Live monitoring session")

    st.success(f"Connected to stream. Monitoring Session: `{session_id}`")

    try:
        frame_count = 0
        for frame_idx, frame in vp.frame_generator(stop_check=lambda: not st.session_state.get("is_monitoring", False)):
            t_start = time.time()
            frame_count += 1

            if max_frames_limit and frame_count >= max_frames_limit:
                st.info(f"Target frame limit reached ({max_frames_limit} frames). Live monitoring session completed.")
                break

            # 1. ByteTrack Object Tracking (detect all 80 COCO classes if enabled)
            try:
                tracks = tracker.track(frame, track_all=detect_all_objects)
            except TypeError:
                # Direct Ultralytics tracking fallback if in-memory ByteTracker class signature is stale
                if detect_all_objects and hasattr(tracker.detector, "model") and tracker.detector.model is not None:
                    try:
                        results = tracker.detector.model.track(
                            source=frame,
                            persist=True,
                            tracker=tracker.tracker_type,
                            conf=tracker.detector.confidence,
                            iou=tracker.detector.iou,
                            imgsz=tracker.detector.imgsz,
                            device=tracker.detector.device,
                            classes=None,
                            verbose=False,
                        )
                        tracks = []
                        if results and results[0].boxes is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                            confidences = results[0].boxes.conf.cpu().numpy()
                            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                            track_ids = (
                                results[0].boxes.id.cpu().numpy().astype(int)
                                if results[0].boxes.id is not None
                                else [None] * len(boxes)
                            )
                            model_names = getattr(tracker.detector.model, "names", {})
                            from core.detector import Detection
                            for bbox, conf, cls_id, tid in zip(boxes, confidences, class_ids, track_ids):
                                cname = model_names.get(int(cls_id), f"class_{cls_id}")
                                tracks.append(
                                    Detection(
                                        class_id=int(cls_id),
                                        class_name=cname,
                                        confidence=float(conf),
                                        bbox=tuple(bbox),
                                        track_id=int(tid) if tid is not None else None,
                                    )
                                )
                    except Exception:
                        tracks = tracker.track(frame)
                else:
                    tracks = tracker.track(frame)

            # 2. Extract detected class counts
            active_class_counts: Dict[str, int] = {}
            for t in tracks:
                cname = t.class_name.capitalize()
                active_class_counts[cname] = active_class_counts.get(cname, 0) + 1

            t_elapsed = max(0.001, time.time() - t_start)
            current_fps = 1.0 / t_elapsed

            # 3. Clean Draw Overlays - Pure object detection without counting line
            annotated = tracker.draw_tracks(frame, tracks, show_trails=True)

            # 4. Write to output video if recording enabled
            if save_live_output:
                vp.write_processed_frame(annotated)

            # 5. Render live frame
            render_video_preview(frame_placeholder, annotated)

            # 6. Render dynamic stats cleanly without duplicate cards
            with stats_placeholder.container():
                st.markdown(f"**Live Feed**: `{source_label}` | **Session**: `{session_id}`")
                st.markdown("<span style='background:#10B98122;color:#10B981;border:1px solid #10B98144;padding:4px 12px;border-radius:20px;font-size:0.85rem;font-weight:600;'>🟢 LIVE MONITORING ACTIVE</span>", unsafe_allow_html=True)

                st.markdown("---")
                stat_cols = st.columns(2)
                with stat_cols[0]:
                    render_metric_card("Total Objects", len(tracks), "Visible in camera", "#3B82F6", "🎯")
                    render_metric_card("Persons", active_class_counts.get("Person", 0), "People in view", "#00D26A", "👤")
                with stat_cols[1]:
                    render_metric_card("Categories", len(active_class_counts), "Unique classes", "#F59E0B", "🏷️")
                    render_metric_card("Live FPS", f"{current_fps:.1f}", "Processing rate", "#06B6D4", "⚡")

                st.markdown("##### 🔍 Detected In Camera")
                if active_class_counts:
                    badges_html = " ".join([
                        f"<span style='display:inline-block;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);padding:5px 12px;border-radius:8px;margin:3px;font-size:0.88rem;font-weight:600;color:#F8FAFC;'>{c}: <b style='color:#38BDF8;'>{cnt}</b></span>"
                        for c, cnt in sorted(active_class_counts.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.caption("No objects currently detected in camera frame.")

    finally:
        vp.close()
        try:
            db.finalize_session(
                session_id=session_id,
                duration_sec=0.0,
                total_vehicles=len(tracks) if 'tracks' in locals() else 0,
                status="completed",
            )
            st.info("Live monitoring stopped. Session recorded in database.")
        except Exception as fin_err:
            logger.warning("Could not finalize live session in database: %s", fin_err)


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

    source_mode = st.radio(
        "Choose Video Input Source:",
        ["📁 Upload Traffic Video File", "🧪 Preloaded Real Traffic Sample"],
        horizontal=True,
        key="video_source_selection_mode",
    )

    selected_video_path: Optional[Path] = None
    original_name: str = ""

    if source_mode == "📁 Upload Traffic Video File":
        uploaded_file = st.file_uploader(
            "Select video file to upload",
            type=["mp4", "avi", "mov", "mkv"],
            help=f"Supported formats: mp4, avi, mov, mkv. Max size: {config.max_upload_size_mb} MB",
            key="user_uploaded_video_file",
        )
        if uploaded_file is not None:
            original_name = uploaded_file.name
            safe_name = sanitize_filename(original_name)
            target_path = UPLOADS_DIR / safe_name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_video_path = target_path
            st.success(f"Uploaded file ready: **{original_name}** ({format_bytes(target_path.stat().st_size)})")
        else:
            # Check for existing uploaded videos in data/uploads/
            existing_uploads = [
                f for f in UPLOADS_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in config.allowed_extensions and f.name != ".gitkeep"
            ]
            if existing_uploads:
                st.markdown(
                    "<p style='color: #94A3B8; font-size: 0.88rem; margin-top: 12px; margin-bottom: 4px;'>Or choose from your previously uploaded videos:</p>",
                    unsafe_allow_html=True,
                )
                prev_choice = st.selectbox(
                    "Select Uploaded Video:",
                    options=[f.name for f in existing_uploads],
                    index=0,
                    key="prev_uploaded_file_select",
                )
                if prev_choice:
                    selected_video_path = UPLOADS_DIR / prev_choice
                    original_name = prev_choice
                    st.info(f"Selected uploaded video: **{original_name}** ({format_bytes(selected_video_path.stat().st_size)})")

    else:
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
        save_annotated = st.checkbox(
            "Save Annotated Output Video to Disk",
            value=False,
            help="Disabled by default: Fast in-memory detection without saving large output video files to disk.",
        )
    with cfg_cols[1]:
        show_live_preview = st.checkbox("Show Preview During Processing", value=True)
    with cfg_cols[2]:
        frame_skip = st.number_input("Process Every Nth Frame", min_value=1, max_value=5, value=1) - 1

    if not save_annotated:
        st.markdown(
            """
            <div style="background: rgba(0, 210, 106, 0.08); border: 1px solid rgba(0, 210, 106, 0.25); border-radius: 8px; padding: 10px 14px; margin: 10px 0 16px 0; color: #34D399; font-size: 0.88rem; font-weight: 500;">
                ⚡ <b>Real-Time Fast Detection Mode:</b> Video upload hote hi instant detect & count hoga — output video disk par save nahi hogi (Storage Saved).
            </div>
            """,
            unsafe_allow_html=True,
        )

    is_proc = st.session_state.get("is_processing", False)
    if is_proc:
        st.warning("⏳ Video processing is currently in progress. Please wait for completion.")
    else:
        if st.button("🚀 Start Video Analysis", type="primary", key="btn_start_video_analysis"):
            st.session_state["is_processing"] = True
            try:
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
            finally:
                st.session_state["is_processing"] = False


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
                cls_cnts = counter.state.class_counts
                try:
                    db.save_snapshot(
                        session_id=session_id,
                        frame_number=frame_idx,
                        active_vehicles=snapshot.active_vehicles,
                        congestion_level=snapshot.congestion_level,
                        cumulative_inbound=snapshot.cumulative_inbound,
                        cumulative_outbound=snapshot.cumulative_outbound,
                        total_crossings=counter.state.total_crossed,
                        current_fps=snapshot.current_fps,
                        cars=cls_cnts.get("car", 0),
                        buses=cls_cnts.get("bus", 0),
                        trucks=cls_cnts.get("truck", 0),
                        motorcycles=cls_cnts.get("motorcycle", 0),
                    )
                except Exception as snap_err:
                    logger.warning("Video snapshot save non-critical error on frame %d: %s", frame_idx, snap_err)

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

    except Exception as e:
        logger.error("Error during video processing in app: %s", e)
        st.error(f"Error during video processing: {e}")
        try:
            if hasattr(db, "update_session_status"):
                db.update_session_status(session_id, status="failed", error_summary=str(e))
        except Exception as stat_err:
            logger.warning("Could not record session failure status: %s", stat_err)
    finally:
        vp.close()
        try:
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
                total_crossings=summary["total_line_crossings"],
                inbound_count=summary["inbound_count"],
                outbound_count=summary["outbound_count"],
                avg_fps=summary["average_fps"],
                output_video_path=str(vp.output_path) if vp.output_path else None,
                status="completed",
            )
        except Exception as fin_err:
            logger.warning("Could not finalize session in database: %s", fin_err)

    # Display final session results
    st.markdown("### 📊 Final Session Analytics")
    render_summary_kpis(summary)

    res_cols = st.columns(2)
    with res_cols[0]:
        st.pyplot(plot_class_distribution(counter.state.class_counts), use_container_width=True)
    with res_cols[1]:
        st.pyplot(plot_directional_traffic(summary["inbound_count"], summary["outbound_count"]), use_container_width=True)

    # In-browser Video Playback
    if save_annotated and vp.output_path and vp.output_path.exists() and vp.output_path.stat().st_size > 0:
        st.markdown("### 🎬 Processed Video Playback")
        st.video(str(vp.output_path))
    elif not save_annotated:
        st.markdown(
            """
            <div style="background: rgba(0, 210, 106, 0.08); border: 1px solid rgba(0, 210, 106, 0.25); border-radius: 10px; padding: 14px 18px; margin: 18px 0; color: #34D399; font-size: 0.92rem;">
                ✅ <b>Live Video Detection & Vehicle Counting Completed:</b> Video was processed and audited in real-time. Output video file was not saved to disk as requested (Storage saved).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Downloads & Exports
    st.markdown("### 📥 Export Reports & Video")
    dl_cols = st.columns(3)

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
            "total_crossings": [summary["total_line_crossings"]],
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
        if save_annotated and vp.output_path and vp.output_path.exists():
            with open(vp.output_path, "rb") as vf:
                video_bytes = vf.read()
            st.download_button(
                label="🎬 Download Annotated Video",
                data=video_bytes,
                file_name=vp.output_path.name,
                mime="video/mp4",
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 12px; text-align: center; color: #94A3B8; font-size: 0.85rem;">
                    ⚡ <b>Fast Memory Mode</b><br>Output video disk par save nahi hui.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with dl_cols[2]:
        if st.button("🔄 Process Another Video", key="btn_reset_analysis"):
            st.session_state["is_processing"] = False
            st.rerun()


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

        st.caption(
            "ℹ️ Note: Congestion classification is an operational heuristic estimated from instantaneous vehicle presence "
            "and bounding-box area density, not a validated physical traffic simulation."
        )

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
    """Render historical session search, filters, detailed inspection, and full CSV export."""
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

        # Detailed Session Drill-down Inspector
        st.markdown("### 🔎 Inspect Individual Session Records")
        inspect_opts = {s["session_id"]: f"{s['session_id']} | {s.get('source_name', 'Unnamed')} | {s.get('start_time', '')}" for s in sessions}
        chosen_sid = st.selectbox(
            "Select Session to Inspect:",
            options=list(inspect_opts.keys()),
            format_func=lambda sid: inspect_opts[sid],
            key="history_inspect_session_select",
        )
        sel_sess = db.get_session(chosen_sid)
        if sel_sess:
            render_summary_kpis(sel_sess)

            # Export individual session CSV button
            sess_df = pd.DataFrame([sel_sess])
            st.download_button(
                label=f"📄 Download Details for {chosen_sid} (CSV)",
                data=sess_df.to_csv(index=False),
                file_name=f"session_{chosen_sid}.csv",
                mime="text/csv",
            )

            snaps = db.get_snapshots_for_session(chosen_sid)
            if snaps:
                with st.expander(f"View {len(snaps)} Temporal Snapshots for {chosen_sid}", expanded=False):
                    st.dataframe(pd.DataFrame(snaps), use_container_width=True, hide_index=True)

        with st.expander("🔍 Interactive Raw Sessions Table", expanded=False):
            df = pd.DataFrame(sessions)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No sessions match the selected filter criteria.")


def render_settings_page(db: DatabaseManager):
    """Render system configuration, hardware status, and database diagnostics."""
    st.markdown("## ⚙️ System Settings & Diagnostics")
    st.markdown(
        "<p style='color: #94A3B8;'>Configure detection parameters, counting line positions, model weights, and inspect database state.</p>",
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

        st.markdown("### Virtual Counting Line Settings")
        st.slider(
            "Default Virtual Line Ratio",
            0.15, 0.85, config.default_line_ratio, 0.05,
            key="settings_line_ratio",
            help="Vertical placement ratio of virtual line relative to frame height (0.0=top, 1.0=bottom)",
        )
        st.number_input(
            "Counting Band Buffer Offset (Pixels)",
            min_value=5, max_value=50, value=config.counting_line_offset_pixels, step=5,
            help="Detection buffer pixel band for crossing boundary evaluation",
        )

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
