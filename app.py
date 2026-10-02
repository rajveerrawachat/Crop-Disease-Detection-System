"""
Crop Disease Detection & Environmental Monitoring System
UI refresh: professional navigation, botanical palette, and simulation/live sensor toggle.

This file preserves the existing project APIs for model inference, sensors,
SQLite persistence, and report generation.
"""
import html
from pathlib import Path

import streamlit as st
from PIL import Image
from datetime import datetime
from model import predict_leaf, CLASS_NAMES
from sensors import (
    get_sensor_data,
    validate_sensor_data,
    analyze_environment,
    list_available_ports,
    SerialManager,
    DEFAULT_PORT,
    DEFAULT_BAUD,
)
from database import (
    init_database,
    save_analysis,
    get_analysis_history,
    delete_analysis,
    get_statistics,
)
from utils import (
    get_confidence_interpretation,
    generate_sample_leaf_image,
    generate_text_report,
)

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Health | Plant Intelligence",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PALETTE = {
    "forest": "#007A33",
    "leaf": "#4CAF50",
    "sage": "#A5D6A7",
    "amber": "#FFCC80",
    "coral": "#FF8965",
}

if "operating_mode" not in st.session_state:
    st.session_state.operating_mode = "Simulation Mode"
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = "Selected leaf"
if "selected_port" not in st.session_state:
    st.session_state.selected_port = DEFAULT_PORT
if "selected_baud" not in st.session_state:
    st.session_state.selected_baud = DEFAULT_BAUD

serial_mgr = SerialManager.get_instance()
init_database()

css_vars = """
      --bg: #f7faf7; --surface: #ffffff; --surface-2: #edf5ee;
      --text: #1b2b20; --muted: #5d7162; --border: #dce8de;
      --primary: #007A33; --primary-strong: #006329; --soft: #e7f3e8;
      --warning-bg: #FFF3DF; --warning-text: #704515;
      --danger-bg: #FFF0EB; --danger-text: #873D2D;
      --shadow: 0 8px 24px rgba(24,70,37,.06);
    """

st.markdown(
    f"""
    <style>
    :root {{ {css_vars} }}
    html, body, [data-testid="stAppViewContainer"] {{
        background: var(--bg);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ display: none; }}
    .block-container {{ max-width: 1440px; padding-top: 1.15rem; padding-bottom: 3rem; }}
    h1, h2, h3, h4, p, label, span, div {{ color: inherit; }}
    h1 {{ letter-spacing: -0.035em; font-weight: 750; }}
    h2, h3 {{ letter-spacing: -0.02em; }}
    .topbar {{
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding: 1rem 1.25rem; background:var(--surface);
        border:1px solid var(--border); border-radius:18px;
        box-shadow:var(--shadow); margin-bottom:1.1rem;
    }}
    .brand {{ display:flex; align-items:center; gap:.75rem; min-width:190px; }}
    .brand-mark {{
        width:42px; height:42px; border-radius:13px; display:flex;
        align-items:center; justify-content:center; background:#e3f2e5;
        color:#007A33; font-size:1.45rem;
    }}
    .brand-name {{ font-size:1.05rem; font-weight:800; color:var(--text); line-height:1.15; }}
    .brand-sub {{ font-size:.76rem; color:var(--muted); margin-top:.2rem; }}
    .eyebrow {{ color:var(--primary); text-transform:uppercase; letter-spacing:.12em;
        font-size:.72rem; font-weight:800; }}
    .muted {{ color:var(--muted); }}
    .hero {{
        padding:1.5rem 1.6rem; border:1px solid var(--border); border-radius:20px;
        background:linear-gradient(120deg, var(--surface) 0%, var(--surface-2) 100%);
        margin:.8rem 0 1.2rem;
    }}
    .hero h1 {{ font-size:clamp(1.65rem,3vw,2.35rem); margin:.3rem 0 .45rem; }}
    .hero p {{ color:var(--muted); margin:0; max-width:780px; }}
    .panel {{
        background:var(--surface); border:1px solid var(--border);
        border-radius:18px; padding:1.1rem 1.2rem; box-shadow:var(--shadow);
        margin-bottom:1rem;
    }}
    .panel-title {{ font-size:1.04rem; font-weight:750; margin:0 0 .25rem; }}
    .panel-caption {{ color:var(--muted); font-size:.86rem; margin-bottom:.8rem; }}
    .status-pill {{
        display:inline-flex; align-items:center; gap:.4rem; border-radius:999px;
        padding:.38rem .72rem; font-size:.78rem; font-weight:750;
        background:var(--soft); color:var(--primary); border:1px solid var(--border);
    }}
    .status-pill.sim {{ background:var(--warning-bg); color:var(--warning-text); }}
    .status-pill.error {{ background:var(--danger-bg); color:var(--danger-text); }}
    .metric-label {{ color:var(--muted); font-size:.82rem; font-weight:650; }}
    .metric-value {{ font-size:1.65rem; font-weight:800; margin-top:.25rem; color:var(--text); }}
    .metric-card {{
        background:var(--surface); border:1px solid var(--border); border-radius:16px;
        padding:1rem; min-height:108px; box-shadow:var(--shadow);
    }}
    .metric-icon {{ font-size:1.25rem; margin-bottom:.45rem; }}
    .result-highlight {{
        background:var(--surface-2); border:1px solid var(--border);
        border-left:5px solid var(--primary); border-radius:14px;
        padding:1rem 1.1rem; margin:.5rem 0 1rem;
    }}
    .small-note {{ color:var(--muted); font-size:.82rem; }}
    .nav-shell {{
        display:flex; align-items:center; gap:.35rem; flex-wrap:wrap;
        background:var(--surface); border:1px solid var(--border);
        border-radius:16px; padding:.45rem; box-shadow:var(--shadow);
        margin:.35rem 0 1rem;
    }}
    .nav-caption {{
        color:var(--muted); font-size:.72rem; font-weight:800;
        letter-spacing:.09em; text-transform:uppercase;
        margin:.15rem 0 .45rem;
    }}
    div.stButton > button, div.stDownloadButton > button {{
        border-radius:11px; font-weight:700; min-height:2.65rem;
        border:1px solid var(--border);
    }}
    /* Navigation buttons are styled as a real horizontal navbar, not radio controls. */
    div[data-testid="stHorizontalBlock"] .stButton > button {{
        transition: background .15s ease, border-color .15s ease;
    }}
    div.stButton > button[kind="primary"] {{
        background:var(--primary); border-color:var(--primary); color:white;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background:var(--primary-strong); border-color:var(--primary-strong); color:white;
    }}
    [data-testid="stMetric"] {{
        background:var(--surface); border:1px solid var(--border);
        border-radius:14px; padding:.8rem 1rem;
    }}
    div[data-testid="stToggle"] {{ width: 100%; }}
    [data-testid="stExpander"] {{
        border:1px solid var(--border); border-radius:13px; overflow:hidden;
    }}
    [data-baseweb="tab-list"] {{ gap:.35rem; }}
    [data-baseweb="tab"] {{ border-radius:10px 10px 0 0; }}
    hr {{ border-color:var(--border); }}
    @media (max-width: 760px) {{
        .block-container {{ padding-left:1rem; padding-right:1rem; }}
        .topbar {{ padding:.8rem; flex-wrap:wrap; }}
        .brand {{ min-width:unset; }}
        .nav-shell {{ padding:.35rem; }}
        .mode-heading {{ text-align:left; margin-top:.35rem; }}

    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# PROFESSIONAL NAVBAR AND OPERATING-MODE TOGGLE
# ---------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

# Two navigation buttons on the left, operating mode grouped on the right.
nav_cols = st.columns([1, 1.2, 1.8], gap="small", vertical_alignment="center")

for col, page_name in zip(nav_cols[:2], ["Dashboard", "Analysis History"]):
    with col:
        if st.button(
            page_name,
            key=f"nav_{page_name.lower().replace(' ', '_')}",
            type="primary" if st.session_state.active_page == page_name else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = page_name
            st.rerun()

with nav_cols[2]:
    st.markdown(
        """
        <div class="mode-heading"></div>
        """,
        unsafe_allow_html=True,
    )

    # Center the complete toggle row beneath the heading.
    left_space, toggle_area, right_space = st.columns(
        [1, 2.4, 1],
        gap="small",
        vertical_alignment="center",
    )
    with toggle_area:
        if "live_sensor_toggle" not in st.session_state:
            st.session_state.live_sensor_toggle = (
                st.session_state.operating_mode == "Live Sensor Mode"
            )

        live_enabled = st.toggle(
            "Live Sensor Input",
            key="live_sensor_toggle",
            help="On: use readings from the connected ESP32. Off: use simulated readings.",
        )

    new_mode = "Live Sensor Mode" if live_enabled else "Simulation Mode"
    if new_mode != st.session_state.operating_mode:
        st.session_state.operating_mode = new_mode
        st.session_state.sensor_data = None
        if new_mode == "Simulation Mode":
            serial_mgr.disconnect()
            st.session_state.sensor_data = get_sensor_data(mode="simulation")
        st.rerun()

# Show a compact, explicit status line so the active mode is never ambiguous.
if st.session_state.operating_mode == "Simulation Mode":
    st.markdown(
        '<span class="status-pill sim">● SIMULATION MODE</span> '
        '<span class="small-note">Synthetic readings · hardware is not being polled.</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-pill">● LIVE SENSOR MODE</span> '
        '<span class="small-note">Uses validated readings from the connected ESP32 only.</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------
def safe_num(value, suffix="", digits=1):
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "Unavailable"


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value">{html.escape(value)}</div>
          <div class="small-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sensor_controls():
    if st.session_state.operating_mode != "Live Sensor Mode":
        if st.button("Refresh simulated readings", key="refresh_sim"):
            st.session_state.sensor_data = get_sensor_data(mode="simulation")
            st.rerun()
        return

    ports = list(list_available_ports())
    if DEFAULT_PORT not in ports:
        ports.append(DEFAULT_PORT)
    if not ports:
        ports = [DEFAULT_PORT]

    control1, control2, control3 = st.columns([2, 1, 1])
    with control1:
        port = st.selectbox(
            "ESP32 serial port",
            ports,
            index=ports.index(st.session_state.selected_port)
            if st.session_state.selected_port in ports else 0,
            key="port_select",
        )
        st.session_state.selected_port = port
    with control2:
        baud = st.selectbox(
            "Baud rate",
            [115200, 9600, 57600, 230400],
            index=[115200, 9600, 57600, 230400].index(st.session_state.selected_baud)
            if st.session_state.selected_baud in [115200, 9600, 57600, 230400] else 0,
            key="baud_select",
        )
        st.session_state.selected_baud = baud
    with control3:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("Connect", type="primary", use_container_width=True):
            success, message = serial_mgr.connect(port, baud)
            if success:
                st.success(f"Serial connection opened for {port}. Waiting for a valid packet.")
            else:
                st.error(f"Could not connect: {message}")
            st.rerun()

    a, b = st.columns(2)
    with a:
        if st.button("Refresh sensor data", use_container_width=True):
            st.session_state.sensor_data = get_sensor_data(
                mode="live", port=st.session_state.selected_port,
                baudrate=st.session_state.selected_baud,
            )
            st.rerun()
    with b:
        if st.button("Disconnect", use_container_width=True):
            serial_mgr.disconnect()
            st.session_state.sensor_data = None
            st.info("ESP32 disconnected.")
            st.rerun()
    st.caption("Hardware: DHT22 → GPIO 4 · Soil moisture AO → GPIO 34 · Serial: USB")


def get_current_sensor_data():
    if st.session_state.operating_mode == "Simulation Mode":
        current = st.session_state.sensor_data
        if not current or current.get("source") != "Simulation":
            current = get_sensor_data(mode="simulation")
            st.session_state.sensor_data = current
        return current or {}
    # Live mode must never fall back to simulation.
    current = get_sensor_data(
        mode="live",
        port=st.session_state.selected_port,
        baudrate=st.session_state.selected_baud,
    )
    st.session_state.sensor_data = current
    return current or {}


# ---------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------
page = st.session_state.active_page

if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Crop monitoring workspace</div>
          <h1>Understand your plant's health.</h1>
          <p>Analyse a leaf image and review environmental readings in one place.
          Sensor data provides context; the image classifier makes the disease-category prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sensor_info = get_current_sensor_data()
    is_sim = st.session_state.operating_mode == "Simulation Mode"
    source = sensor_info.get("source", "Unavailable")
    temp = sensor_info.get("temperature_c")
    humidity = sensor_info.get("humidity_pct")
    moisture = sensor_info.get("soil_moisture_pct")
    conn_state = sensor_info.get("connection_state", "Disconnected")

    if is_sim:
        st.markdown('<span class="status-pill sim">SIMULATED ENVIRONMENTAL DATA</span>', unsafe_allow_html=True)
    elif source == "ESP32" and conn_state == "Connected":
        st.markdown(
            f'<span class="status-pill">ESP32 CONNECTED · {html.escape(st.session_state.selected_port)}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill error">SENSOR DATA UNAVAILABLE</span>',
            unsafe_allow_html=True,
        )
        st.warning(
            "Live readings are not currently available. Connect the ESP32, select its COM port, "
            "and refresh the sensor data. No simulated values are substituted in Live Sensor Mode."
        )

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        metric_card("Air temperature" + (" · simulated" if is_sim else ""),
                    safe_num(temp, " °C"), "DHT22 · GPIO 4")
    with m2:
        metric_card("Air humidity" + (" · simulated" if is_sim else ""),
                    safe_num(humidity, " % RH"), "DHT22 · GPIO 4")
    with m3:
        metric_card("Soil moisture" + (" · simulated" if is_sim else ""),
                    safe_num(moisture, " %"), "Analog sensor · GPIO 34")

    if st.session_state.operating_mode == "Live Sensor Mode":
        with st.expander("Sensor connection settings", expanded=False):
            show_sensor_controls()
    else:
        with st.expander("Simulation settings", expanded=False):
            st.write("Simulation mode uses synthetic values for demonstration only.")
            show_sensor_controls()

    st.markdown("### Leaf analysis")
    upload_col, preview_col = st.columns([1, 1], gap="large")
    with upload_col:
        st.markdown('<div class="panel-title">Add a leaf photograph</div>', unsafe_allow_html=True)
        st.caption("Use a clear image of one leaf. Supported formats: JPG, JPEG, PNG.")
        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="leaf_uploader",
        )
        sample_a, sample_b = st.columns(2)
        with sample_a:
            if st.button("Use healthy sample", use_container_width=True):
                st.session_state.current_image = generate_sample_leaf_image(plant="Tomato", healthy=True)
                st.session_state.image_filename = "Healthy sample"
                st.rerun()
        with sample_b:
            if st.button("Use disease sample", use_container_width=True):
                st.session_state.current_image = generate_sample_leaf_image(plant="Tomato", healthy=False)
                st.session_state.image_filename = "Disease sample"
                st.rerun()
        if uploaded is not None:
            try:
                st.session_state.current_image = Image.open(uploaded).convert("RGB")
                # Keep the filename internally for persistence, but do not display local paths.
                st.session_state.image_filename = Path(uploaded.name).name
            except Exception:
                st.error("This image could not be opened. Please choose a valid JPG or PNG image.")

    with preview_col:
        st.markdown('<div class="panel-title">Image preview</div>', unsafe_allow_html=True)
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, use_container_width=True)
            st.caption("Selected leaf image")
        else:
            st.info("Your selected leaf image will appear here.")

    can_analyze = st.session_state.current_image is not None
    if st.button(
        "Analyse leaf",
        type="primary",
        use_container_width=True,
        disabled=not can_analyze,
        key="analyse_leaf",
    ):
        with st.spinner("Analysing the leaf image…"):
            try:
                prediction = predict_leaf(st.session_state.current_image, top_k=5)

                if st.session_state.operating_mode == "Simulation Mode":
                    snapshot = get_sensor_data(mode="simulation")
                    sensor_source = "Simulation"
                else:
                    snapshot = get_sensor_data(
                        mode="live",
                        port=st.session_state.selected_port,
                        baudrate=st.session_state.selected_baud,
                    )
                    sensor_source = snapshot.get("source", "Unavailable")

                # Do not misrepresent missing live sensor data as a successful hardware capture.
                if st.session_state.operating_mode == "Live Sensor Mode":
                    valid_live = (
                        sensor_source == "ESP32"
                        and snapshot.get("temperature_c") is not None
                        and snapshot.get("soil_moisture_pct") is not None
                    )
                    if not valid_live:
                        st.warning(
                            "The leaf prediction can still be reviewed, but a valid live sensor snapshot "
                            "was unavailable. The result will be saved with sensor source 'Unavailable'."
                        )
                        sensor_source = "Unavailable"

                c_temp = snapshot.get("temperature_c") if sensor_source != "Unavailable" else None
                c_hum = snapshot.get("humidity_pct") if sensor_source != "Unavailable" else None
                c_moist = snapshot.get("soil_moisture_pct") if sensor_source != "Unavailable" else None
                observations, summary = analyze_environment(c_temp, c_moist, c_hum)

                entry = {
                    "image_name": st.session_state.image_filename,
                    "plant": prediction["plant"],
                    "predicted_condition": prediction["condition"],
                    "predicted_class": prediction["predicted_class"],
                    "confidence": prediction["confidence"],
                    "temperature_c": c_temp,
                    "humidity_pct": c_hum,
                    "soil_moisture_pct": c_moist,
                    "sensor_source": sensor_source,
                    "top_predictions": prediction.get("top_predictions", []),
                    "environmental_observations": observations,
                    "environmental_summary": summary,
                    "image_status": prediction.get("image_status"),
                }
                record_id = save_analysis(entry)
                entry["id"] = record_id
                st.session_state.last_analysis = entry
                st.success("Analysis completed and saved to history.")
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")

    if st.session_state.last_analysis:
        result = st.session_state.last_analysis
        st.markdown("---")
        st.markdown("### Latest result")
        result_left, result_right = st.columns([1.1, 0.9], gap="large")
        with result_left:
            st.markdown('<div class="panel-title">Prediction</div>', unsafe_allow_html=True)
            condition = str(result.get("predicted_condition", "Unknown"))
            if "healthy" in condition.lower():
                st.markdown('<span class="status-pill">HEALTHY CLASS PREDICTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-pill sim">POTENTIAL DISEASE CLASS PREDICTED</span>', unsafe_allow_html=True)
            st.markdown(f"<div class='result-highlight'><div class='eyebrow'>Predicted crop</div><h2>{html.escape(str(result.get('plant', 'Unknown')))}</h2><div><b>{html.escape(condition)}</b></div></div>", unsafe_allow_html=True)
            confidence = float(result.get("confidence", 0.0))
            st.write(f"**Model confidence:** {confidence * 100:.1f}%")
            st.progress(max(0.0, min(1.0, confidence)))
            st.caption(get_confidence_interpretation(confidence))
            candidates = result.get("top_predictions", [])
            if candidates:
                with st.expander("Alternative predictions", expanded=False):
                    for candidate in candidates:
                        readable = candidate.get("readable") or f"{candidate.get('plant', '')} — {candidate.get('condition', '')}"
                        score = float(candidate.get("confidence", 0.0))
                        st.write(f"{readable} · {score * 100:.1f}%")
                        st.progress(max(0.0, min(1.0, score)))
        with result_right:
            st.markdown('<div class="panel-title">Environmental snapshot</div>', unsafe_allow_html=True)
            st.write(f"**Temperature:** {safe_num(result.get('temperature_c'), ' °C')}")
            st.write(f"**Air humidity:** {safe_num(result.get('humidity_pct'), ' % RH')}")
            st.write(f"**Soil moisture:** {safe_num(result.get('soil_moisture_pct'), ' %')}")
            result_source = result.get("sensor_source", "Unavailable")
            if result_source == "ESP32":
                st.markdown('<span class="status-pill">SOURCE: ESP32 HARDWARE</span>', unsafe_allow_html=True)
            elif result_source == "Simulation":
                st.markdown('<span class="status-pill sim">SOURCE: SIMULATION</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-pill error">SOURCE: UNAVAILABLE</span>', unsafe_allow_html=True)
            st.markdown("**Environmental context**")
            for observation in result.get("environmental_observations", []):
                st.write(f"• {observation}")
            st.caption("Environmental readings provide context; they do not independently confirm a plant disease.")
            report = generate_text_report(result)
            st.download_button(
                "Download analysis report",
                data=report,
                file_name=f"crop_health_report_{result.get('id', 'latest')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ---------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------
elif page == "Analysis History":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Your records</div>
          <h1>Analysis history</h1>
          <p>Review previous predictions, environmental snapshots, and exported reports.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stats = get_statistics()
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Total analyses", stats.get("total_analyses", 0))
    with s2:
        st.metric("Healthy-class predictions", stats.get("healthy_count", 0))
    with s3:
        st.metric("Potential disease predictions", stats.get("diseased_count", 0))

    records = get_analysis_history(limit=50)
    if not records:
        st.info("No analyses have been saved yet. Run a leaf analysis from the Dashboard.")
    else:
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            query = st.text_input("Search by crop or predicted condition", key="history_query")
        with filter_col2:
            source_filter = st.selectbox(
                "Data source",
                ["All sources", "ESP32 hardware", "Simulation", "Unavailable"],
                key="history_source",
            )
        filtered = records
        if query:
            filtered = [
                row for row in filtered
                if query.lower() in str(row.get("plant", "")).lower()
                or query.lower() in str(row.get("predicted_condition", "")).lower()
            ]
        source_map = {
            "ESP32 hardware": "ESP32",
            "Simulation": "Simulation",
            "Unavailable": "Unavailable",
        }
        if source_filter in source_map:
            filtered = [row for row in filtered if row.get("sensor_source", "Unavailable") == source_map[source_filter]]
        st.caption(f"{len(filtered)} records shown")
        for row in filtered:
            record_id = row.get("id")
            timestamp = row.get("timestamp", "Unknown date")
            plant = row.get("plant", "Unknown crop")
            condition = row.get("predicted_condition", "Unknown condition")
            confidence = float(row.get("confidence") or 0) * 100
            source = row.get("sensor_source", "Unavailable")
            with st.expander(f"{timestamp}  ·  {plant} — {condition}  ·  {confidence:.1f}%"):
                detail_col, sensor_col = st.columns([1.2, 1])
                with detail_col:
                    st.write(f"**Crop:** {plant}")
                    st.write(f"**Prediction:** {condition}")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    candidates = row.get("top_predictions") or []
                    if candidates:
                        st.markdown("**Alternative predictions**")
                        for candidate in candidates:
                            readable = candidate.get("readable") or f"{candidate.get('plant', '')} — {candidate.get('condition', '')}"
                            st.write(f"• {readable} ({float(candidate.get('confidence', 0))*100:.1f}%)")
                with sensor_col:
                    st.write(f"**Temperature:** {safe_num(row.get('temperature_c'), ' °C')}")
                    st.write(f"**Air humidity:** {safe_num(row.get('humidity_pct'), ' % RH')}")
                    st.write(f"**Soil moisture:** {safe_num(row.get('soil_moisture_pct'), ' %')}")
                    st.write(f"**Source:** {source}")
                    for observation in row.get("environmental_observations") or []:
                        st.write(f"• {observation}")
                st.download_button(
                    "Export this report",
                    data=generate_text_report(row),
                    file_name=f"crop_health_report_{record_id}.txt",
                    mime="text/plain",
                    key=f"history_export_{record_id}",
                )
                if st.button("Delete record", key=f"history_delete_{record_id}"):
                    delete_analysis(record_id)
                    st.rerun()

st.markdown("---")
st.markdown(
    '<div class="small-note"></div>',
    unsafe_allow_html=True,
)