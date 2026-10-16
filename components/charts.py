"""
Matplotlib visualization components for AI Traffic Monitoring Dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from config.settings import CLASS_COLORS_HEX


def _apply_dark_style(fig, ax):
    """Apply consistent modern dark theme to matplotlib figure and axes."""
    fig.patch.set_facecolor("#151922")
    ax.set_facecolor("#1B212D")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.8)
    ax.grid(color="#2D3748", linestyle="--", linewidth=0.5, alpha=0.7)


def plot_class_distribution(class_counts: Dict[str, int]) -> plt.Figure:
    """
    Plot vehicle and pedestrian class breakdown as a modern bar chart.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    _apply_dark_style(fig, ax)

    classes = [c.capitalize() for c in class_counts.keys()]
    raw_keys = list(class_counts.keys())
    counts = [class_counts[k] for k in raw_keys]
    colors = [CLASS_COLORS_HEX.get(k.lower(), "#3B82F6") for k in raw_keys]

    bars = ax.bar(classes, counts, color=colors, width=0.55, edgecolor="#334155", linewidth=1)

    # Add count labels on top of bars
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#F8FAFC",
            )

    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Detected Category Distribution", fontsize=11, fontweight="bold", pad=12)
    max_count = max(counts) if counts and max(counts) > 0 else 5
    ax.set_ylim(0, max_count * 1.25)
    fig.tight_layout()
    return fig


def plot_directional_traffic(inbound: int, outbound: int) -> plt.Figure:
    """
    Plot horizontal directional comparison (Inbound vs Outbound).
    """
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
    _apply_dark_style(fig, ax)

    categories = ["Outbound", "Inbound"]
    values = [outbound, inbound]
    colors = ["#F59E0B", "#10B981"]

    bars = ax.barh(categories, values, color=colors, height=0.45, edgecolor="#334155")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f" {int(w)}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#F8FAFC",
        )

    ax.set_xlabel("Vehicles Crossed", fontsize=10)
    ax.set_title("Directional Traffic Flow", fontsize=11, fontweight="bold", pad=10)
    max_val = max(values) if values and max(values) > 0 else 5
    ax.set_xlim(0, max_val * 1.3)
    fig.tight_layout()
    return fig


def plot_traffic_trend(snapshots: List[Any]) -> Optional[plt.Figure]:
    """
    Plot temporal trend of active vehicles and cumulative flow over frames.
    """
    if not snapshots or len(snapshots) < 2:
        return None

    frames = [s.frame_number if hasattr(s, "frame_number") else s.get("frame_number", i) for i, s in enumerate(snapshots)]
    active = [s.active_vehicles if hasattr(s, "active_vehicles") else s.get("active_vehicles", 0) for s in snapshots]
    inbound = [s.cumulative_inbound if hasattr(s, "cumulative_inbound") else s.get("cumulative_inbound", 0) for s in snapshots]
    outbound = [s.cumulative_outbound if hasattr(s, "cumulative_outbound") else s.get("cumulative_outbound", 0) for s in snapshots]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=100)
    _apply_dark_style(fig, ax)

    ax.plot(frames, active, label="Active Vehicles", color="#00D26A", linewidth=2.0)
    ax.plot(frames, inbound, label="Cumulative Inbound", color="#3B82F6", linewidth=1.8, linestyle="--")
    ax.plot(frames, outbound, label="Cumulative Outbound", color="#F59E0B", linewidth=1.8, linestyle=":")

    ax.set_xlabel("Video Frame", fontsize=10)
    ax.set_ylabel("Vehicle Count", fontsize=10)
    ax.set_title("Traffic Flow & Activity Trend", fontsize=11, fontweight="bold", pad=12)

    legend = ax.legend(loc="upper left", facecolor="#1B212D", edgecolor="#334155", fontsize=9)
    for text in legend.get_texts():
        text.set_color("#CBD5E1")

    fig.tight_layout()
    return fig


def plot_fps_performance(snapshots: List[Any]) -> Optional[plt.Figure]:
    """
    Plot processing FPS throughput over frame history.
    """
    if not snapshots or len(snapshots) < 2:
        return None

    frames = [s.frame_number if hasattr(s, "frame_number") else s.get("frame_number", i) for i, s in enumerate(snapshots)]
    fps_vals = [s.current_fps if hasattr(s, "current_fps") else s.get("current_fps", 0.0) for s in snapshots]

    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
    _apply_dark_style(fig, ax)

    ax.plot(frames, fps_vals, label="Processing FPS", color="#06B6D4", linewidth=1.8)
    avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0
    ax.axhline(avg_fps, color="#EF4444", linestyle="--", linewidth=1.2, label=f"Mean: {avg_fps:.1f} FPS")

    ax.set_xlabel("Video Frame", fontsize=9)
    ax.set_ylabel("FPS", fontsize=9)
    ax.set_title("Real-Time Inference Performance", fontsize=10, fontweight="bold")

    legend = ax.legend(loc="upper right", facecolor="#1B212D", edgecolor="#334155", fontsize=8)
    for text in legend.get_texts():
        text.set_color("#CBD5E1")

    fig.tight_layout()
    return fig


def plot_sessions_history(sessions: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """
    Plot total vehicles and directional flow across recent processing sessions.
    """
    if not sessions:
        return None

    # Filter to sessions with valid data and take up to the 8 most recent
    valid_sessions = [s for s in sessions if s.get("total_vehicles", 0) > 0 or s.get("inbound_count", 0) > 0 or s.get("outbound_count", 0) > 0]
    if not valid_sessions:
        valid_sessions = sessions[:8]
    else:
        valid_sessions = valid_sessions[:8]

    # Chronological left to right
    valid_sessions = list(reversed(valid_sessions))

    labels = [
        (s.get("source_name") or s.get("session_id", ""))[:14]
        for s in valid_sessions
    ]
    inbounds = [s.get("inbound_count", 0) for s in valid_sessions]
    outbounds = [s.get("outbound_count", 0) for s in valid_sessions]

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    _apply_dark_style(fig, ax)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, inbounds, width, label="Inbound Flow", color="#3B82F6", edgecolor="#334155")
    ax.bar(x + width / 2, outbounds, width, label="Outbound Flow", color="#F59E0B", edgecolor="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Line Crossings", fontsize=9)
    ax.set_title("Recent Sessions Flow Comparison", fontsize=10, fontweight="bold", pad=10)

    legend = ax.legend(loc="upper left", facecolor="#1B212D", edgecolor="#334155", fontsize=8)
    for text in legend.get_texts():
        text.set_color("#CBD5E1")

    fig.tight_layout()
    return fig
