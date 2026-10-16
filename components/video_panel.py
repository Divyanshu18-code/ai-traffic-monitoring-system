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

    # Downscale high-resolution frames (e.g. 4K) for smooth UI streaming
    h, w = frame_bgr.shape[:2]
    max_preview_w = 960
    if w > max_preview_w:
        scale = max_preview_w / w
        preview_h = int(h * scale)
        frame_to_show = cv2.resize(frame_bgr, (max_preview_w, preview_h), interpolation=cv2.INTER_AREA)
    else:
        frame_to_show = frame_bgr

    # Convert BGR to RGB for Streamlit rendering
    frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
    placeholder.image(frame_rgb, caption=caption, use_container_width=True)
