"""
Video display panel and playback controls for Streamlit Dashboard.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import streamlit as st


def render_video_preview(
    placeholder: st.delta_generator.DeltaGenerator,
    frame_bgr: np.ndarray,
    caption: Optional[str] = None,
) -> None:
    """
    Render a single video frame into a Streamlit placeholder efficiently.

    Args:
        placeholder: Streamlit empty() placeholder.
        frame_bgr: Frame in OpenCV BGR format.
        caption: Optional frame caption.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return

    # Convert BGR to RGB for Streamlit rendering
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    placeholder.image(frame_rgb, caption=caption, use_container_width=True)
