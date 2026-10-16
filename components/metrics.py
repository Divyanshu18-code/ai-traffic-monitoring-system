"""
Metrics and KPI rendering components for Streamlit Dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def render_metric_card(
    label: str,
    value: Any,
    subtext: str = "",
    color: str = "#00D26A",
    icon: str = "🚗",
) -> None:
    """Render a modern glassmorphic metric card with custom styling."""
    html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 36, 48, 0.7), rgba(18, 22, 31, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid {color};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">
                {label}
            </span>
            <span style="font-size: 1.1rem;">{icon}</span>
        </div>
        <div style="font-size: 1.85rem; font-weight: 700; color: #F8FAFC; line-height: 1.2;">
            {value}
        </div>
        {f'<div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_congestion_badge(level: str) -> None:
    """Render a traffic congestion status pill badge."""
    lvl = (level or "LOW").upper()
    colors = {
        "LOW": {"bg": "rgba(16, 185, 129, 0.15)", "border": "#10B981", "text": "#34D399"},
        "MODERATE": {"bg": "rgba(245, 158, 11, 0.15)", "border": "#F59E0B", "text": "#FBBF24"},
        "HIGH": {"bg": "rgba(239, 68, 68, 0.15)", "border": "#EF4444", "text": "#F87171"},
    }
    c = colors.get(lvl, colors["LOW"])

    html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {c['bg']};
        border: 1px solid {c['border']};
        border-radius: 20px;
        padding: 4px 14px;
        color: {c['text']};
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    ">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: {c['border']};"></span>
        CONGESTION: {lvl}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_summary_kpis(summary: Dict[str, Any]) -> None:
    """Render a responsive row of KPI cards based on analytics summary."""
    cols = st.columns(5)

    with cols[0]:
        total_veh = summary.get("total_unique_vehicles", summary.get("total_vehicles", 0))
        render_metric_card("Total Vehicles", total_veh, "Unique tracked objects", "#00D26A", "🚘")

    with cols[1]:
        inbound = summary.get("inbound_count", 0)
        render_metric_card("Inbound Flow", inbound, "Vehicles entering line", "#3B82F6", "⬇️")

    with cols[2]:
        outbound = summary.get("outbound_count", 0)
        render_metric_card("Outbound Flow", outbound, "Vehicles exiting line", "#F59E0B", "⬆️")

    with cols[3]:
        peds = summary.get("total_unique_pedestrians", summary.get("pedestrians", 0))
        render_metric_card("Pedestrians", peds, "Observed foot traffic", "#8B5CF6", "🚶")

    with cols[4]:
        fps = summary.get("average_fps", summary.get("avg_fps", 0.0))
        render_metric_card("Throughput", f"{fps:.1f} FPS", "Inference & tracking rate", "#06B6D4", "⚡")


def render_sessions_html_table(sessions: list[dict[str, Any]]) -> None:
    """
    Render a clean, high-contrast, modern dark table for past audit sessions.
    """
    if not sessions:
        st.info("No recorded sessions found.")
        return

    source_badges = {
        "video_file": ("🎬 Video File", "background: rgba(6, 182, 212, 0.15); color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.4);"),
        "webcam": ("📹 Webcam", "background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4);"),
        "ip_camera": ("🌐 IP Camera", "background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4);"),
    }

    rows = []
    for s in sessions:
        sid = s.get("session_id", "—")
        src_type = s.get("source_type", "video_file")
        badge_label, badge_style = source_badges.get(src_type, (src_type, "color: #94A3B8;"))
        src_name = s.get("source_name", "—")
        start_time = s.get("start_time", "—")
        total_veh = s.get("total_vehicles", 0)
        in_cnt = s.get("inbound_count", 0)
        out_cnt = s.get("outbound_count", 0)
        avg_fps = s.get("avg_fps", 0.0)

        row = (
            f'<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.06);">'
            f'<td style="padding: 12px 14px; font-family: monospace; font-size: 0.85rem; color: #38BDF8; font-weight: 600;">{sid}</td>'
            f'<td style="padding: 12px 14px;"><span style="display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; {badge_style}">{badge_label}</span></td>'
            f'<td style="padding: 12px 14px; font-size: 0.88rem; color: #E2E8F0; font-weight: 500;">{src_name}</td>'
            f'<td style="padding: 12px 14px; font-size: 0.82rem; color: #94A3B8;">{start_time}</td>'
            f'<td style="padding: 12px 14px; font-size: 0.95rem; font-weight: 700; color: #00D26A; text-align: center;">{total_veh}</td>'
            f'<td style="padding: 12px 14px; font-size: 0.88rem; color: #38BDF8; text-align: center;">{in_cnt}</td>'
            f'<td style="padding: 12px 14px; font-size: 0.88rem; color: #FBBF24; text-align: center;">{out_cnt}</td>'
            f'<td style="padding: 12px 14px; font-size: 0.85rem; color: #CBD5E1; text-align: right;">{avg_fps:.1f} FPS</td>'
            f'</tr>'
        )
        rows.append(row)

    tbody_content = "".join(rows)

    table_html = (
        '<div style="border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; overflow-x: auto; background: #141A26; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); margin: 12px 0 20px 0;">'
        '<table style="width: 100%; border-collapse: collapse; text-align: left;">'
        '<thead>'
        '<tr style="background: #1A2232; border-bottom: 2px solid rgba(255, 255, 255, 0.1);">'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Session ID</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Source</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Name</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8;">Timestamp</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; text-align: center;">Total Vehicles</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; text-align: center;">Inbound</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; text-align: center;">Outbound</th>'
        '<th style="padding: 12px 14px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; text-align: right;">Throughput</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{tbody_content}</tbody>'
        '</table>'
        '</div>'
    )

    if hasattr(st, "html"):
        st.html(table_html)
    else:
        st.markdown(table_html, unsafe_allow_html=True)

