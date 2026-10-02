"""
Crop Disease Detection and Environmental Monitoring System
Streamlit Web Application with Real-Time Hardware Telemetry & Serial Monitor
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st

from model import predict_leaf, CLASS_NAMES
from sensors import (
    get_sensor_data,
    validate_sensor_data,
    analyze_environment,
    list_available_ports,
    SerialManager,
    DEFAULT_PORT,
    DEFAULT_BAUD,
    DEFAULT_STALE_TIMEOUT
)
from database import (
    init_database,
    save_analysis,
    get_analysis_history,
    delete_analysis,
    get_statistics
)
from utils import (
    get_confidence_interpretation,
    generate_sample_leaf_image,
    generate_text_report
)

# ------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------

st.set_page_config(
    page_title="Crop Health & Real-Time Hardware Monitor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global styles */
    .main {
        background-color: #fcfdfc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Card containers */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-healthy {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-disease {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-sensor-esp {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-sensor-sim {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-sensor-warn {
        background-color: #fff7ed;
        color: #c2410c;
        border: 1px solid #fed7aa;
    }
    .badge-sensor-err {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Hardware Serial Monitor Console */
    .serial-terminal {
        background-color: #0e1117;
        color: #00ff66;
        font-family: "Courier New", Courier, monospace;
        padding: 14px;
        border-radius: 8px;
        height: 220px;
        overflow-y: auto;
        border: 1px solid #1e293b;
        font-size: 0.88rem;
        line-height: 1.4;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
    }
    .terminal-line {
        margin: 2px 0;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .terminal-err {
        color: #ff4d4d;
    }
    .terminal-info {
        color: #38bdf8;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize database schema if not present
init_database()

# Session state initialization
if "operating_mode" not in st.session_state:
    st.session_state.operating_mode = "Simulation Mode"
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = "sample.jpg"
if "selected_port" not in st.session_state:
    st.session_state.selected_port = DEFAULT_PORT
if "selected_baud" not in st.session_state:
    st.session_state.selected_baud = DEFAULT_BAUD

serial_mgr = SerialManager.get_instance()

# ------------------------------------------------------------
# SIDEBAR NAVIGATION & OPERATING MODE SELECTOR
# ------------------------------------------------------------

with st.sidebar:
    st.title("🌱 Plant Health Monitor")
    st.caption("College Capstone ML + IoT Hardware Project")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard & Live Monitor", "Analysis History", "System Architecture & C++ Code"],
        index=0
    )

    st.markdown("---")
    st.subheader("⚙️ System Operating Mode")

    # MUTUALLY EXCLUSIVE TWO-MODE SELECTOR
    mode_options = ["Simulation Mode", "Live Sensor Mode"]
    current_mode_index = mode_options.index(st.session_state.operating_mode) if st.session_state.operating_mode in mode_options else 0

    chosen_mode = st.radio(
        "Select Operating Mode:",
        options=mode_options,
        index=current_mode_index,
        help="Select Simulation Mode for offline demonstration or Live Sensor Mode to read physical ESP32 measurements."
    )

    # Handle mode change safely
    if chosen_mode != st.session_state.operating_mode:
        st.session_state.operating_mode = chosen_mode
        if chosen_mode == "Simulation Mode":
            # Safely disconnect physical hardware when switching to simulation
            serial_mgr.disconnect()
            st.session_state.sensor_data = get_sensor_data(mode="simulation")
        else:
            # Switched to Live Sensor Mode: clear simulated readings immediately
            st.session_state.sensor_data = None
        st.rerun()

    st.markdown("---")

    # Mode-Specific Sidebar Controls
    if st.session_state.operating_mode == "Live Sensor Mode":
        st.subheader("🔌 Physical ESP32 Serial Settings")
        
        available_ports = list_available_ports()
        port_choices = list(available_ports)
        if DEFAULT_PORT not in port_choices:
            port_choices.append(DEFAULT_PORT)
        
        port_idx = port_choices.index(st.session_state.selected_port) if st.session_state.selected_port in port_choices else 0
        sel_port = st.selectbox("COM Port:", options=port_choices, index=port_idx)
        st.session_state.selected_port = sel_port

        baud_choices = [115200, 9600, 57600, 230400]
        baud_idx = baud_choices.index(st.session_state.selected_baud) if st.session_state.selected_baud in baud_choices else 0
        sel_baud = st.selectbox("Baud Rate:", options=baud_choices, index=baud_idx)
        st.session_state.selected_baud = sel_baud

        st.caption("Wiring: DHT22 -> GPIO 4 | HW-080 -> GPIO 34")

        col_con1, col_con2 = st.columns(2)
        with col_con1:
            if st.button("🔌 Connect", use_container_width=True):
                success, msg = serial_mgr.connect(sel_port, sel_baud)
                if success:
                    st.success(f"Connected to {sel_port}")
                else:
                    st.error(f"Connect failed: {msg}")
                st.rerun()

        with col_con2:
            if st.button("❌ Disconnect", use_container_width=True):
                serial_mgr.disconnect()
                st.info("Serial disconnected.")
                st.session_state.sensor_data = None
                st.rerun()

        col_poll1, col_poll2 = st.columns(2)
        with col_poll1:
            if st.button("🔄 Poll / Refresh", use_container_width=True):
                st.session_state.sensor_data = get_sensor_data(
                    mode="live",
                    port=st.session_state.selected_port,
                    baudrate=st.session_state.selected_baud
                )
                st.rerun()
        with col_poll2:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                serial_mgr.clear_logs()
                st.rerun()

    else:
        # Simulation Mode Sidebar Info
        st.subheader("🧪 Simulation Controls")
        st.info(
            "**Simulation Active**: Generating synthetic microclimate telemetry. "
            "All readings are explicitly tagged as SIMULATED. Physical hardware is not polled."
        )
        if st.button("🔄 Generate Fresh Simulation Data", use_container_width=True):
            st.session_state.sensor_data = get_sensor_data(mode="simulation")
            st.rerun()

    st.markdown("---")
    st.caption("ESP32 (GPIO 4 DHT22 + GPIO 34 HW-080) • EfficientNet-B4 • SQLite")


# ------------------------------------------------------------
# TAB 1: DASHBOARD & LIVE MONITOR
# ------------------------------------------------------------

if page == "Dashboard & Live Monitor":

    st.header("Crop Disease Detection & Real-Time Environmental Telemetry")
    st.markdown(
        "Multimodal crop diagnostics integrating **EfficientNet-B4 computer vision classification** "
        "with **ESP32 physical microclimate telemetry** over USB Serial."
    )
    st.markdown("---")

    # Fetch latest sensor data according to the selected mode
    if st.session_state.operating_mode == "Simulation Mode":
        if st.session_state.sensor_data is None or st.session_state.sensor_data.get("source") != "Simulation":
            st.session_state.sensor_data = get_sensor_data(mode="simulation")
    else:
        # Live Sensor Mode: Fetch from SerialManager without falling back to simulation
        st.session_state.sensor_data = get_sensor_data(
            mode="live",
            port=st.session_state.selected_port,
            baudrate=st.session_state.selected_baud
        )

    sensor_info = st.session_state.sensor_data or {}

    # Two Main Columns: (Left: Leaf Input & Camera, Right: Telemetry & Serial Terminal)
    col_img, col_sensor = st.columns([1, 1.1], gap="large")

    # -----------------------------
    # LEFT COLUMN: Image Input
    # -----------------------------
    with col_img:
        st.markdown('<div class="section-title">🌿 Leaf Image Input</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image of a single crop leaf (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )

        col_sample1, col_sample2 = st.columns(2)
        with col_sample1:
            if st.button("Use Sample Leaf (Healthy)", use_container_width=True):
                st.session_state.current_image = generate_sample_leaf_image(plant="Tomato", healthy=True)
                st.session_state.image_filename = "sample_tomato_healthy.jpg"
        with col_sample2:
            if st.button("Use Sample Leaf (Diseased)", use_container_width=True):
                st.session_state.current_image = generate_sample_leaf_image(plant="Tomato", healthy=False)
                st.session_state.image_filename = "sample_tomato_early_blight.jpg"

        if uploaded_file is not None:
            try:
                st.session_state.current_image = Image.open(uploaded_file)
                st.session_state.image_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Failed to read image file: {e}")

        if st.session_state.current_image is not None:
            st.image(
                st.session_state.current_image,
                caption=f"Selected Leaf: {st.session_state.image_filename}",
                use_container_width=True
            )
        else:
            st.info("Please upload a leaf photograph or select a sample leaf to begin analysis.")

    # -----------------------------
    # RIGHT COLUMN: Telemetry & Serial Monitor
    # -----------------------------
    with col_sensor:
        st.markdown('<div class="section-title">📡 Real-Time Environmental Telemetry</div>', unsafe_allow_html=True)

        is_sim_mode = (st.session_state.operating_mode == "Simulation Mode")
        source = sensor_info.get("source", "Unavailable")
        temp = sensor_info.get("temperature_c")
        hum = sensor_info.get("humidity_pct")
        moist = sensor_info.get("soil_moisture_pct")
        status_text = sensor_info.get("status", "Unknown")
        conn_state = sensor_info.get("connection_state", "Disconnected")
        data_age = sensor_info.get("data_age")
        last_ts = sensor_info.get("timestamp")

        # Visual Mode & Connection Status Badges
        if is_sim_mode:
            badge_html = '<span class="status-badge badge-sensor-sim">🟡 OPERATING MODE: SIMULATION (OFFLINE DEMO)</span>'
            st.markdown(badge_html, unsafe_allow_html=True)
            st.caption("Synthetic microclimate readings generated for offline testing without hardware.")
        else:
            # Live Sensor Mode: honest connection state display
            if conn_state == "Connected" and source == "ESP32":
                badge_html = f'<span class="status-badge badge-sensor-esp">🟢 LIVE SENSOR MODE: Connected ({st.session_state.selected_port} @ {st.session_state.selected_baud} baud)</span>'
            elif conn_state == "Connecting":
                badge_html = f'<span class="status-badge badge-sensor-warn">🟡 LIVE SENSOR MODE: Connecting / Awaiting Data ({st.session_state.selected_port})</span>'
            elif conn_state == "Stale":
                badge_html = f'<span class="status-badge badge-sensor-warn">🟠 LIVE SENSOR MODE: Stale Data ({data_age}s since last packet)</span>'
            elif conn_state == "Error":
                badge_html = f'<span class="status-badge badge-sensor-err">🔴 LIVE SENSOR MODE: Hardware Error ({sensor_info.get("error", "Unknown error")})</span>'
            else:
                badge_html = f'<span class="status-badge badge-sensor-err">🔴 LIVE SENSOR MODE: Disconnected ({st.session_state.selected_port})</span>'
            
            st.markdown(badge_html, unsafe_allow_html=True)

            # Metadata strip for physical sensor connection
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.caption(f"**Port**: `{st.session_state.selected_port}` ({st.session_state.selected_baud} baud)")
            with col_meta2:
                st.caption(f"**Last Packet**: `{last_ts or 'None'}`")
            with col_meta3:
                age_str = f"{data_age:.1f}s ago" if data_age is not None else "N/A"
                st.caption(f"**Data Age**: `{age_str}`")

        # Telemetry Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        sim_tag = " [SIMULATED]" if is_sim_mode else ""
        with m_col1:
            st.metric(
                label=f"🌡️ Temperature{sim_tag}",
                value=f"{temp:.1f} °C" if temp is not None else "Unavailable",
                help="Ambient air temperature from DHT22 on GPIO 4"
            )
        with m_col2:
            st.metric(
                label=f"💧 Air Humidity{sim_tag}",
                value=f"{hum:.1f} % RH" if hum is not None else "Unavailable",
                help="Relative air humidity from DHT22 on GPIO 4"
            )
        with m_col3:
            st.metric(
                label=f"🌱 Soil Moisture{sim_tag}",
                value=f"{moist:.1f} %" if moist is not None else "Unavailable",
                help="Volumetric soil moisture percentage from HW-080 on GPIO 34"
            )

        # Validation warnings
        if not is_sim_mode and source == "Unavailable":
            st.warning("⚠️ Live sensor telemetry is currently unavailable. Ensure the ESP32 is flashed, connected to USB, and the correct COM port is opened.")
        else:
            sensor_warnings = validate_sensor_data(temp, moist, hum)
            if sensor_warnings and source != "Unavailable":
                for w in sensor_warnings:
                    st.warning(f"⚠️ {w}")

        # Hardware Serial Terminal Console
        st.markdown("##### 💻 Hardware Serial Monitor Output")
        logs = serial_mgr.get_logs() if not is_sim_mode else []
        if is_sim_mode:
            terminal_html = (
                '<div class="serial-terminal">'
                '<div class="terminal-line terminal-info">[Simulation Mode Active: Hardware serial polling is disabled.]</div>'
                f'<div class="terminal-line">SIM_TX: {sensor_info.get("raw_line", "")}</div>'
                '</div>'
            )
        elif not logs:
            terminal_html = (
                '<div class="serial-terminal">'
                f'<div class="terminal-line terminal-info">[Serial Monitor initialized for {st.session_state.selected_port}]</div>'
                '<div class="terminal-line">[Waiting for incoming JSON telemetry packets from ESP32...]</div>'
                '</div>'
            )
        else:
            lines_html = []
            for log in logs:
                css_cls = "terminal-err" if log["type"] == "err" else ("terminal-info" if log["type"] == "info" else "terminal-line")
                lines_html.append(f'<div class="{css_cls}">[{log["timestamp"]}] {log["text"]}</div>')
            terminal_html = f'<div class="serial-terminal">{"".join(lines_html)}</div>'

        st.markdown(terminal_html, unsafe_allow_html=True)

        # Environmental Context Observations
        obs, env_summary = analyze_environment(temp, moist, hum)
        with st.expander("Microclimate Context & Environmental Assessment", expanded=True):
            for ob in obs:
                st.markdown(f"• {ob}")
            st.caption(f"**Context**: {env_summary}")

    st.markdown("---")

    # -----------------------------
    # ANALYZE BUTTON
    # -----------------------------
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_clicked = st.button(
            "🔍 Analyze Leaf & Correlate Environmental Telemetry",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.current_image is None)
        )

    if analyze_clicked and st.session_state.current_image is not None:
        with st.spinner("Classifying leaf with EfficientNet-B4 & capturing telemetry snapshot..."):
            try:
                # 1. Run ML leaf prediction
                prediction = predict_leaf(st.session_state.current_image, top_k=5)

                # 2. Capture latest environmental telemetry snapshot based on active mode
                if st.session_state.operating_mode == "Simulation Mode":
                    snapshot_sensors = get_sensor_data(mode="simulation")
                    c_source = "Simulation"
                else:
                    snapshot_sensors = get_sensor_data(
                        mode="live",
                        port=st.session_state.selected_port,
                        baudrate=st.session_state.selected_baud
                    )
                    c_source = snapshot_sensors.get("source", "Unavailable")

                c_temp = snapshot_sensors.get("temperature_c")
                c_hum = snapshot_sensors.get("humidity_pct")
                c_moist = snapshot_sensors.get("soil_moisture_pct")
                c_obs, c_sum = analyze_environment(c_temp, c_moist, c_hum)

                # 3. Assemble structured analysis object
                analysis_entry = {
                    "image_name": st.session_state.image_filename,
                    "plant": prediction["plant"],
                    "predicted_condition": prediction["condition"],
                    "predicted_class": prediction["predicted_class"],
                    "confidence": prediction["confidence"],
                    "temperature_c": c_temp,
                    "humidity_pct": c_hum,
                    "soil_moisture_pct": c_moist,
                    "sensor_source": c_source,
                    "top_predictions": prediction["top_predictions"],
                    "environmental_observations": c_obs,
                    "environmental_summary": c_sum,
                    "image_status": prediction["image_status"]
                }

                # 4. Save to SQLite database
                inserted_id = save_analysis(analysis_entry)
                analysis_entry["id"] = inserted_id
                st.session_state.last_analysis = analysis_entry

                st.success(f"Analysis completed and persisted to SQLite database (Record #{inserted_id})!")

            except Exception as e:
                st.error(f"Inference or analysis error encountered: {e}")

    # -----------------------------
    # DIAGNOSTIC RESULTS DISPLAY
    # -----------------------------
    if st.session_state.last_analysis is not None:
        res = st.session_state.last_analysis
        st.markdown("### Latest Diagnostic Report")

        res_col1, res_col2 = st.columns([1.2, 1], gap="large")

        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"#### Target Plant: **{res['plant']}**")
            st.markdown(f"**Predicted Condition**: `{res['predicted_condition']}`")

            conf_pct = res["confidence"] * 100
            st.markdown(f"**Model Confidence**: **{conf_pct:.2f}%**")
            st.progress(min(res["confidence"], 1.0))

            # Status badge
            if "healthy" in res["predicted_condition"].lower():
                status_badge = '<span class="status-badge badge-healthy">Healthy Leaf Class</span>'
            else:
                status_badge = '<span class="status-badge badge-disease">Potential Disease Detected</span>'
            st.markdown(f"**Diagnostic Status**: {status_badge}", unsafe_allow_html=True)

            conf_explanation = get_confidence_interpretation(res["confidence"])
            st.caption(f"ℹ️ *Confidence Interpretation*: {conf_explanation}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Top-5 Predictions
            st.markdown("##### Candidate Predictions (Top 5)")
            for pred in res.get("top_predictions", []):
                p_label = pred.get("readable") or f"{pred.get('plant')} — {pred.get('condition')}"
                p_conf = pred.get("confidence", 0.0) * 100
                st.write(f"• **{p_label}**: `{p_conf:.2f}%`")
                st.progress(min(pred.get("confidence", 0.0), 1.0))

        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Hardware Telemetry Snapshot")

            r_temp = res.get("temperature_c")
            r_hum = res.get("humidity_pct")
            r_moist = res.get("soil_moisture_pct")
            r_source = res.get("sensor_source", "Unavailable")

            st.write(f"**Temperature (DHT22)**: {f'{r_temp:.1f} °C' if r_temp is not None else 'Unavailable'}")
            st.write(f"**Air Humidity (DHT22)**: {f'{r_hum:.1f} % RH' if r_hum is not None else 'Unavailable'}")
            st.write(f"**Soil Moisture (HW-080)**: {f'{r_moist:.1f} %' if r_moist is not None else 'Unavailable'}")
            
            # Explicit source tag
            if r_source == "ESP32":
                st.markdown("**Data Source**: <span class='status-badge badge-sensor-esp'>ESP32 Hardware (Validated)</span>", unsafe_allow_html=True)
            elif r_source == "Simulation":
                st.markdown("**Data Source**: <span class='status-badge badge-sensor-sim'>Simulation (Synthetic Demonstration)</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Data Source**: <span class='status-badge badge-sensor-err'>Unavailable (Disconnected / Stale)</span>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Correlated Microclimate Observations**:")
            for ob in res.get("environmental_observations", []):
                st.write(f"- {ob}")

            st.info(
                "**Academic Context Reminder**:\n"
                "Sensor readings provide microclimate context for image-based disease prediction. "
                "They should not be interpreted as biological proof of disease."
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Export Report Button
            report_text = generate_text_report(res)
            st.download_button(
                label="📥 Download Full Health & Telemetry Report (.txt)",
                data=report_text,
                file_name=f"crop_health_report_{res.get('id', 'latest')}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ------------------------------------------------------------
# TAB 2: ANALYSIS HISTORY
# ------------------------------------------------------------

elif page == "Analysis History":

    st.header("Analysis History (SQLite Persistence)")
    st.markdown("Browse, inspect, and export previous plant disease analyses stored in `crop_disease.db`.")
    st.markdown("---")

    # Overview Analytics Summary
    stats = get_statistics()
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("Total Analyses Logged", stats["total_analyses"])
    with s_col2:
        st.metric("Healthy Plant Cases", stats["healthy_count"])
    with s_col3:
        st.metric("Diseased Plant Cases", stats["diseased_count"])

    st.markdown("---")

    records = get_analysis_history(limit=50)

    if not records:
        st.info("No analysis records found in SQLite database yet. Perform an analysis on the Dashboard to populate history.")
    else:
        # Search & Filter
        fil_col1, fil_col2 = st.columns([2, 1])
        with fil_col1:
            search_query = st.text_input("Filter records by plant or disease condition:", "")
        with fil_col2:
            source_filter = st.selectbox("Filter by Sensor Source:", ["All Sources", "ESP32 Only", "Simulation Only"])

        filtered = records
        if search_query:
            filtered = [
                r for r in filtered
                if search_query.lower() in r["plant"].lower() or search_query.lower() in r["predicted_condition"].lower()
            ]
        if source_filter == "ESP32 Only":
            filtered = [r for r in filtered if r.get("sensor_source") == "ESP32"]
        elif source_filter == "Simulation Only":
            filtered = [r for r in filtered if r.get("sensor_source") == "Simulation"]

        st.caption(f"Showing {len(filtered)} of {len(records)} recorded analyses.")

        for row in filtered:
            record_id = row["id"]
            timestamp = row["timestamp"]
            plant = row["plant"]
            condition = row["predicted_condition"]
            confidence = row["confidence"] * 100
            temp = row["temperature_c"]
            hum = row.get("humidity_pct")
            moist = row["soil_moisture_pct"]
            source = row.get("sensor_source", "Unavailable")

            with st.expander(f"#{record_id} | {timestamp} — {plant}: {condition} ({confidence:.1f}%) [{source}]"):
                det_col1, det_col2 = st.columns([1.5, 1])

                with det_col1:
                    st.markdown(f"**Target Plant**: {plant}")
                    st.markdown(f"**Condition**: `{condition}`")
                    st.markdown(f"**Confidence**: `{confidence:.2f}%`")
                    st.markdown(f"**Image Filename**: `{row.get('image_name', 'N/A')}`")

                    if row.get("top_predictions"):
                        st.markdown("**Top Alternative Predictions**:")
                        for p in row["top_predictions"]:
                            p_readable = p.get("readable") or f"{p.get('plant')} — {p.get('condition')}"
                            st.write(f"- {p_readable} (`{p.get('confidence', 0.0) * 100:.2f}%`)")

                with det_col2:
                    st.markdown("**Hardware Telemetry**:")
                    st.write(f"- Temp (DHT22): {f'{temp:.1f} °C' if temp is not None else 'N/A'}")
                    st.write(f"- Air Humidity (DHT22): {f'{hum:.1f} %' if hum is not None else 'N/A'}")
                    st.write(f"- Soil Moisture (HW-080): {f'{moist:.1f} %' if moist is not None else 'N/A'}")
                    
                    if source == "ESP32":
                        st.markdown("- Source: <span class='status-badge badge-sensor-esp'>ESP32 Hardware</span>", unsafe_allow_html=True)
                    elif source == "Simulation":
                        st.markdown("- Source: <span class='status-badge badge-sensor-sim'>Simulation</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- Source: `{source}`")

                    if row.get("environmental_observations"):
                        st.markdown("**Observations**:")
                        for ob in row["environmental_observations"]:
                            st.write(f"• {ob}")

                    # Export & Delete Actions
                    rep_data = generate_text_report(row)
                    st.download_button(
                        label="📄 Export Report",
                        data=rep_data,
                        file_name=f"report_{record_id}.txt",
                        mime="text/plain",
                        key=f"dl_{record_id}"
                    )

                    if st.button("🗑️ Delete Record", key=f"del_{record_id}"):
                        delete_analysis(record_id)
                        st.warning(f"Record #{record_id} deleted.")
                        st.rerun()


# ------------------------------------------------------------
# TAB 3: SYSTEM ARCHITECTURE & C++ CODE
# ------------------------------------------------------------

elif page == "System Architecture & C++ Code":

    st.header("System Specifications & Microcontroller C++ Code")
    st.markdown(
        "Technical reference documentation for college project demonstration and viva evaluation."
    )
    st.markdown("---")

    spec_col1, spec_col2 = st.columns(2)

    with spec_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Machine Learning Subsystem")
        st.markdown("""
        - **Architecture**: `EfficientNet-B4` (Custom 2-layer classifier head)
        - **Base Weights**: ImageNet pretraining + PlantVillage fine-tuning
        - **Checkpoint**: `Khawajaa/plant-disease-detector` (`best_model.pth`)
        - **Classes Covered**: 38 plant disease & healthy classes
        - **Input Dimensions**: $3 \\times 224 \\times 224$ (RGB)
        - **Preprocessing**: Resize(256) $\\rightarrow$ CenterCrop(224) $\\rightarrow$ ImageNet Normalization
        - **Inference Engine**: PyTorch
        - **Top-5 Evaluation**: Multi-class softmax ranking with calibrated confidence
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with spec_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🔌 Hardware Pinout & Serial Protocol")
        st.markdown("""
        - **Microcontroller**: ESP32 Dev Module
        - **Sensors & Wiring**:
          1. **DHT22** (Temp & Air Humidity): Connected to **GPIO 4** (3.3V VCC)
          2. **HW-080** (Soil Moisture Analog): Connected to **ADC GPIO 34** (3.3V VCC)
        - **Voltage Caution**: All sensor inputs must use 3.3V. **Never apply 5V directly to ESP32 pins!**
        - **Serial Configuration**: **115200 Baud** over USB Serial
        - **Data Packet Format**: Newline-delimited JSON
          ```json
          {"temperature": 25.4, "humidity": 60.1, "soil_moisture": 45.2}
          ```
        - **Error Packet Format**:
          ```json
          {"error": "DHT22_read_failed", "soil_moisture": 45.2}
          ```
        - **Database Storage**: Embedded SQLite (`crop_disease.db`)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔬 ESP32 / Arduino Microcontroller C++ Sketch Code")
    st.markdown("Flash this sketch using Arduino IDE to stream telemetry from DHT22 (GPIO 4) and Soil Moisture (GPIO 34):")

    # Read sketch directly from firmware file if present
    sketch_path = Path(__file__).resolve().parent / "firmware" / "esp32_dht22_soil" / "esp32_dht22_soil.ino"
    if sketch_path.exists():
        sketch_code = sketch_path.read_text(encoding="utf-8")
    else:
        sketch_code = "// Firmware sketch file not found."

    st.code(sketch_code, language="cpp")

    st.markdown("---")
    st.markdown("#### 📋 38 Supported PlantVillage Classes")
    with st.expander("Click to view all 38 classes"):
        c_cols = st.columns(3)
        for idx, cname in enumerate(CLASS_NAMES):
            c_cols[idx % 3].write(f"{idx + 1}. `{cname}`")