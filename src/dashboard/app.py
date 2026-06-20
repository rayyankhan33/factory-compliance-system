"""
Operations Dashboard — Module 5.

Three required views, as tabs:
  A. Live Feed Monitor      -- pick a clip, watch it, see its latest status + any active alert
  B. Alert Timeline Stream  -- chronological feed of all events, severity-colored, auto-refreshing
  C. Historical Log & Export -- filterable table over all events with CSV/JSON export

Run with:
    streamlit run src/dashboard/app.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from db import database  # noqa: E402
from video_discovery import discover_video_clips, clip_id_for_path, resolve_clip_id_root  # noqa: E402

st.set_page_config(page_title="Factory Compliance Dashboard", layout="wide")

SEVERITY_COLORS = {
    "LOW": "#4caf50",
    "MEDIUM": "#ffb300",
    "HIGH": "#fb8c00",
    "CRITICAL": "#e53935",
}


def severity_badge(severity):
    color = SEVERITY_COLORS.get(severity, "#9e9e9e")
    return f'<span style="background-color:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:600;">{severity}</span>'


st.title("🏭 Factory Compliance & Alert Escalation System")

database.init_db()

tab_live, tab_alerts, tab_history = st.tabs(
    ["📹 Live Feed Monitor", "🚨 Alert Timeline Stream", "📊 Historical Log & Export"]
)

# ---------------------------------------------------------------------------
# A. Live Feed Monitor
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Live Feed Monitor")
    data_dir = ROOT / "data"
    clip_id_root = resolve_clip_id_root(data_dir, data_dir)
    clip_paths = discover_video_clips(data_dir) if data_dir.exists() else []
    clip_lookup = {clip_id_for_path(p, data_dir, clip_id_root=clip_id_root): p for p in clip_paths}

    if not clip_paths:
        st.info("No clips found in `data/`. Add video clips and run `src/pipeline.py` to populate this view.")
    else:
        clip_names = sorted(clip_lookup)
        selected = st.selectbox("Select a clip", clip_names)
        selected_path = clip_lookup[selected]

        col_video, col_status = st.columns([2, 1])
        with col_video:
            st.video(str(selected_path))

        with col_status:
            st.markdown("**Status**")
            status_by_clip = database.get_latest_status_per_clip()
            status = status_by_clip.get(selected)
            if status:
                st.markdown(
                    f"Latest event: **{status['behavior_class']}**  \n"
                    f"Severity: {severity_badge(status['severity'])}  \n"
                    f"At: {status['timestamp']}",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("No events recorded yet for this clip.")

            active_alerts = [a for a in database.get_active_alerts() if a["clip_id"] == selected]
            if active_alerts:
                for a in active_alerts:
                    st.error(f"⚠️ {a['message']}")
            else:
                st.success("No active alerts for this clip.")

        st.caption(
            "Run `python src/pipeline.py --backend yolo_world --data-dir data --reset` to (re)process "
            "clips and populate live status."
        )

# ---------------------------------------------------------------------------
# B. Alert Timeline Stream
# ---------------------------------------------------------------------------
with tab_alerts:
    st.subheader("Alert Timeline Stream")
    auto_refresh = st.checkbox("Auto-refresh every 5s", value=False)

    events = database.query_events(limit=200)
    if not events:
        st.info("No events logged yet. Run the pipeline against some clips first.")
    else:
        for e in events:
            color = SEVERITY_COLORS.get(e["severity"], "#9e9e9e")
            st.markdown(
                f"""
                <div style="border-left:5px solid {color}; padding:8px 14px; margin-bottom:6px; background-color:rgba(128,128,128,0.07); border-radius:4px;">
                  <b>{e['timestamp']}</b> &nbsp; {severity_badge(e['severity'])} &nbsp;
                  <b>{e['behavior_class']}</b> &nbsp;|&nbsp; clip: {e['clip_id']} &nbsp;|&nbsp; zone: {e['zone']} &nbsp;|&nbsp; ref: {e['policy_rule_ref']}<br/>
                  <span style="opacity:0.85;">{e['event_description']}</span> &nbsp;
                  <i>({e['escalation_action']})</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ---------------------------------------------------------------------------
# C. Historical Log & Export
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Historical Log & Export")

    col1, col2, col3 = st.columns(3)
    with col1:
        sev_filter = st.multiselect("Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    with col2:
        class_filter = st.multiselect(
            "Behavior class",
            options=[0, 1, 2, 3],
            format_func=lambda c: {
                0: "Safe Walkway Violation", 1: "Unauthorized Intervention",
                2: "Opened Panel Cover", 3: "Carrying Overload with Forklift",
            }[c],
        )
    with col3:
        date_range = st.date_input("Date range", value=())

    start_date = end_date = None
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date = date_range[0].strftime("%Y-%m-%dT00:00:00Z")
        end_date = date_range[1].strftime("%Y-%m-%dT23:59:59Z")

    filtered = database.query_events(
        start_date=start_date, end_date=end_date,
        severities=sev_filter or None, class_ids=class_filter or None,
        limit=5000,
    )

    if not filtered:
        st.info("No events match the current filters.")
    else:
        df = pd.DataFrame(filtered)
        display_cols = ["timestamp", "clip_id", "zone", "behavior_class", "policy_rule_ref",
                         "event_description", "severity", "escalation_action"]
        st.dataframe(df[display_cols], width="stretch", height=420)

        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button(
                "⬇️ Export filtered results as CSV",
                df[display_cols].to_csv(index=False),
                file_name="compliance_export.csv",
                mime="text/csv",
            )
        with col_json:
            st.download_button(
                "⬇️ Export filtered results as JSON",
                df[display_cols].to_json(orient="records", indent=2),
                file_name="compliance_export.json",
                mime="application/json",
            )
