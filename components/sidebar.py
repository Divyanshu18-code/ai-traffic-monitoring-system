"""
Sidebar navigation and global controls component for Streamlit Dashboard.
"""

from __future__ import annotations

from typing import Tuple

import streamlit as st
import torch

from config.settings import get_config


def render_sidebar() -> Tuple[str, float, float]:
    """
    Render application sidebar navigation and control widgets.

    Returns:
        (selected_page, confidence_threshold, line_ratio)
    """
    config = get_config()

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 20px 0; text-align: center;">
                <h2 style="color: #F8FAFC; margin: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em;">
                    AI Traffic Monitor
                </h2>
                <div style="font-size: 0.75rem; color: #00D26A; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;">
                    Real-Time Computer Vision
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Page Navigation
        pages = [
            "Overview",
            "Live Monitoring",
            "Video Analysis",
            "Traffic Analytics",
            "History",
            "Settings",
        ]

        icons = {
            "Overview": "📊",
            "Live Monitoring": "📹",
            "Video Analysis": "🎬",
            "Traffic Analytics": "📈",
            "History": "📜",
            "Settings": "⚙️",
        }

        # Direct session state binding for instant 1-click navigation
        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "Overview"

        selected_page = st.radio(
            "Navigation",
            options=pages,
            format_func=lambda p: f"{icons.get(p, '•')}  {p}",
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Global Quick Controls
        st.markdown(
            "<div style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; margin-bottom: 8px;'>QUICK CONTROLS</div>",
            unsafe_allow_html=True,
        )

        confidence_val = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(st.session_state.get("conf_threshold", config.confidence_threshold)),
            step=0.05,
            help="Minimum confidence score required for object detection",
        )
        st.session_state["conf_threshold"] = confidence_val

        line_ratio_val = st.slider(
            "Counting Line Position",
            min_value=0.15,
            max_value=0.85,
            value=float(st.session_state.get("line_ratio", config.default_line_ratio)),
            step=0.05,
            help="Vertical position of virtual counting line relative to frame height",
        )
        st.session_state["line_ratio"] = line_ratio_val

        st.markdown("---")

        # Hardware & Engine Status Badge
        cuda_avail = torch.cuda.is_available()
        device_label = "CUDA (GPU)" if cuda_avail else "CPU Engine"
        device_color = "#10B981" if cuda_avail else "#3B82F6"

        st.markdown(
            f"""
            <div style="
                background: #1B212D;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px;
                font-size: 0.78rem;
                color: #CBD5E1;
            ">
                <div style="font-weight: 600; color: #94A3B8; margin-bottom: 6px;">SYSTEM STATUS</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Execution Device:</span>
                    <span style="color: {device_color}; font-weight: 700;">{device_label}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Model:</span>
                    <span style="color: #F8FAFC; font-weight: 600;">{config.default_model_name}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Tracker:</span>
                    <span style="color: #F8FAFC; font-weight: 600;">ByteTrack</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page, confidence_val, line_ratio_val
