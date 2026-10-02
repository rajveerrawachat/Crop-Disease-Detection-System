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
    DEFAULT_PORT,
    DEFAULT_BAUD
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

# Professional Custom CSS: Clean, modern agricultural dashboard + Dark Serial Terminal
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
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
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
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-sensor-sim {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
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

# Initialize database
init_database()

# Session state initialization
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = "sample.jpg"
if "serial_logs" not in st.session_state:
    st.session_state.serial_logs = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# Helper function to append serial log entry
def append_serial_log(message: str, log_type: str = "out"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"timestamp": ts, "text": message, "type": log_type}
    st.session_state.serial_logs.append(entry)
    # Keep last 50 log lines
    if len(st.session_state.serial_logs) > 50:
        st.session_state.serial_logs.pop(0)


# ------------------------------------------------------------
# SIDEBAR NAVIGATION & HARDWARE CONTROLS
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
    st.subheader("🔌 Hardware Serial Settings")

    sensor_mode_choice = st.selectbox(
        "Sensor Source Mode",
        options=["Auto-Detect (ESP32 / Sim Fallback)", "ESP32 Hardware Only", "Simulation Mode"],
        index=0
    )

    mode_map = {
        "Auto-Detect (ESP32 / Sim Fallback)": "auto",
        "ESP32 Hardware Only": "esp32",
        "Simulation Mode": "simulation"
    }
    active_mode = mode_map[sensor_mode_choice]

    available_ports = list_available_ports()
    selected_port = DEFAULT_PORT
    selected_baud = DEFAULT_BAUD

    if active_mode in ["auto", "esp32"]:
        port_options = available_ports if available_ports else [DEFAULT_PORT]
        selected_port = st.selectbox("Serial COM Port", options=port_options, index=0)
        selected_baud = st.selectbox("Baud Rate", options=[115200, 9600, 57600, 230400], index=0)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Read Hardware"):
            with st.spinner("Polling ESP32 Serial Port..."):
                s_data = get_sensor_data(
                    mode=active_mode,
                    port=selected_port,
                    baudrate=selected_baud
                )
                st.session_state.sensor_data = s_data
                if s_data.get("raw_line"):
                    append_serial_log(f"RX: {s_data['raw_line']}", "out")
                elif s_data.get("error"):
                    append_serial_log(f"ERR: {s_data['error']}", "err")

    with col_btn2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.serial_logs = []

    st.markdown("---")
    st.caption("ESP32 • DHT22 • HW-080 • EfficientNet-B4 • SQLite")


# ------------------------------------------------------------
# TAB 1: DASHBOARD & LIVE HARDWARE MONITOR
# ------------------------------------------------------------

if page == "Dashboard & Live Monitor":

    st.header("Crop Disease Detection & Real-Time Environmental Telemetry")
    st.markdown(
        "Interactive hardware-integrated dashboard combining **computer vision disease classification** "
        "with **real-time physical sensor telemetry streaming** over ESP32 USB Serial."
    )
    st.markdown("---")

    # Fetch initial or updated sensor reading
    if st.session_state.sensor_data is None:
        s_data = get_sensor_data(mode=active_mode, port=selected_port, baudrate=selected_baud)
        st.session_state.sensor_data = s_data
        if s_data.get("raw_line"):
            append_serial_log(f"RX: {s_data['raw_line']}", "out")

    sensor_info = st.session_state.sensor_data

    # Layout: Two Main Columns (Left: Image Upload & Inference, Right: Hardware Telemetry & Serial Monitor)
    col_img, col_sensor = st.columns([1, 1.1], gap="large")

    # -----------------------------
    # LEFT: Leaf Image Upload & Camera Input
    # -----------------------------
    with col_img:
        st.markdown('<div class="section-title">🌿 Leaf Image Input</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image of a single crop leaf",
            type=["jpg", "jpeg", "png"]
        )

        col_sample1, col_sample2 = st.columns(2)
        with col_sample1:
            if st.button("Use Sample Leaf (Healthy)"):
                st.session_state.current_image = generate_sample_leaf_image(plant="Tomato", healthy=True)
                st.session_state.image_filename = "sample_tomato_healthy.jpg"
        with col_sample2:
            if st.button("Use Sample Leaf (Diseased)"):
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
    # RIGHT: Hardware Telemetry & Live Serial Monitor
    # -----------------------------
    with col_sensor:
        st.markdown('<div class="section-title">📡 Real-Time Hardware Telemetry</div>', unsafe_allow_html=True)

        source = sensor_info.get("source", "Unavailable")
        temp = sensor_info.get("temperature_c")
        hum = sensor_info.get("humidity_pct")
        moist = sensor_info.get("soil_moisture_pct")
        status_text = sensor_info.get("status", "Unknown")

        # Connection Status Badge
        if source == "ESP32":
            badge_html = f'<span class="status-badge badge-sensor-esp">🟢 ESP32 Hardware Connected ({selected_port} @ {selected_baud} baud)</span>'
        elif source == "Simulation":
            badge_html = f'<span class="status-badge badge-sensor-sim">🟡 Simulation Mode Active (Hardware disconnected)</span>'
        else:
            badge_html = f'<span class="status-badge badge-sensor-err">🔴 Hardware Disconnected ({status_text})</span>'

        st.markdown(badge_html, unsafe_allow_html=True)
        st.write("")

        # Live Real-Time Sensor Telemetry Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="🌡️ Temperature",
                value=f"{temp:.1f} °C" if temp is not None else "N/A",
                help="Ambient air temperature from DHT22 on PIN 4"
            )
        with m_col2:
            st.metric(
                label="💧 Air Humidity",
                value=f"{hum:.1f} %" if hum is not None else "N/A",
                help="Relative air humidity percentage from DHT22 on PIN 4"
            )
        with m_col3:
            st.metric(
                label="🌱 Soil Moisture",
                value=f"{moist:.1f} %" if moist is not None else "N/A",
                help="Volumetric soil moisture percentage from HW-080 on PIN 34"
            )

        # Validation warnings
        sensor_warnings = validate_sensor_data(temp, moist, hum)
        if sensor_warnings and source != "Unavailable":
            for w in sensor_warnings:
                st.warning(f"⚠️ {w}")

        # Live Hardware Serial Monitor Output
        st.markdown("##### 💻 Hardware Serial Monitor Output")
        
        # Build terminal HTML
        if not st.session_state.serial_logs:
            terminal_html = '<div class="serial-terminal"><div class="terminal-line terminal-info">[System initialized - Waiting for incoming serial packets...]</div></div>'
        else:
            lines_html = []
            for log in st.session_state.serial_logs:
                css_cls = "terminal-err" if log["type"] == "err" else "terminal-line"
                lines_html.append(f'<div class="{css_cls}">[{log["timestamp"]}] {log["text"]}</div>')
            terminal_html = f'<div class="serial-terminal">{"".join(lines_html)}</div>'

        st.markdown(terminal_html, unsafe_allow_html=True)

        # Environmental Context Observations
        obs, env_summary = analyze_environment(temp, moist, hum)
        with st.expander("Microclimate Analysis Details", expanded=True):
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
            "🔍 Analyze Leaf & Correlate Hardware Telemetry",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.current_image is None)
        )

    if analyze_clicked and st.session_state.current_image is not None:
        with st.spinner("Processing leaf image with EfficientNet-B4 & capturing telemetry snapshot..."):
            try:
                # 1. Run ML leaf prediction
                prediction = predict_leaf(st.session_state.current_image, top_k=5)

                # 2. Capture latest environmental telemetry snapshot
                current_sensors = st.session_state.sensor_data or get_sensor_data(mode=active_mode, port=selected_port, baudrate=selected_baud)
                c_temp = current_sensors.get("temperature_c")
                c_hum = current_sensors.get("humidity_pct")
                c_moist = current_sensors.get("soil_moisture_pct")
                c_source = current_sensors.get("source", "Unavailable")
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

                st.success(f"Analysis completed and saved to SQLite database (Record #{inserted_id})!")

            except Exception as e:
                st.error(f"Inference error encountered: {e}")

    # -----------------------------
    # ANALYSIS RESULTS SECTION
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
            st.write(f"**Air Humidity (DHT22)**: {f'{r_hum:.1f} %' if r_hum is not None else 'Unavailable'}")
            st.write(f"**Soil Moisture (HW-080)**: {f'{r_moist:.1f} %' if r_moist is not None else 'Unavailable'}")
            st.write(f"**Data Source**: `{r_source}`")

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
        search_query = st.text_input("Filter records by plant or disease condition:", "")
        filtered = [
            r for r in records
            if search_query.lower() in r["plant"].lower() or search_query.lower() in r["predicted_condition"].lower()
        ] if search_query else records

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
            source = row["sensor_source"]

            is_healthy = "healthy" in condition.lower()

            with st.expander(f"#{record_id} | {timestamp} — {plant}: {condition} ({confidence:.1f}%)"):
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
                    st.write(f"- Source: `{source}`")

                    if row.get("environmental_observations"):
                        st.markdown("**Observations**:")
                        for ob in row["environmental_observations"]:
                            st.write(f"• {ob}")

                    # Actions
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
        "Technical reference documentation for hardware configuration, circuit pinouts, and serial protocol."
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
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with spec_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🔌 Hardware Pinout & Serial Protocol")
        st.markdown("""
        - **Microcontroller**: ESP32 / Arduino Board
        - **Sensors & Wiring**:
          1. **DHT22** (Temp & Air Humidity): Connected to **GPIO 4**
          2. **HW-080** (Soil Moisture Analog): Connected to **ADC GPIO 34**
        - **Serial Configuration**: **115200 Baud** over USB Serial
        - **Data Packet Format**: Newline-delimited JSON
          ```json
          {"temperature": 25.4, "humidity": 60.2, "soil_moisture": 45.0}
          ```
        - **Error Packet Format**:
          ```json
          {"error": "DHT22_read_failed"}
          ```
        - **Database Storage**: Embedded SQLite (`crop_disease.db`)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔬 ESP32 / Arduino Microcontroller C++ Sketch Code")
    st.markdown("This exact C++ sketch runs on your hardware, reading the DHT22 and HW-080 sensors and transmitting telemetry over serial:")

    st.code("""
#include <Arduino.h>
#include "DHT.h"

// DHT22 Pin & Type
const int DHT_PIN = 4;
#define DHT_TYPE DHT22

// HW-080 Soil Moisture Sensor Pin
const int SOIL_PIN = 34;

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();

  delay(2000);
}

void loop() {

  // Read DHT22
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Read HW-080 analog output
  int raw_soil = analogRead(SOIL_PIN);

  // Convert to approximate percentage
  float soil_moisture = map(raw_soil, 4095, 1500, 0, 100);
  soil_moisture = constrain(soil_moisture, 0.0, 100.0);

  // Check DHT22
  if (isnan(temperature) || isnan(humidity)) {

    Serial.println("{\\"error\\":\\"DHT22_read_failed\\"}");

  } else {

    // Send JSON
    Serial.print("{\\"temperature\\":");
    Serial.print(temperature, 1);

    Serial.print(",\\"humidity\\":");
    Serial.print(humidity, 1);

    Serial.print(",\\"soil_moisture\\":");
    Serial.print(soil_moisture, 1);

    Serial.println("}");
  }

  delay(2000);
}
    """, language="cpp")

    st.markdown("---")
    st.markdown("#### 📋 38 Supported PlantVillage Classes")
    with st.expander("Click to view all 38 classes"):
        c_cols = st.columns(3)
        for idx, cname in enumerate(CLASS_NAMES):
            c_cols[idx % 3].write(f"{idx + 1}. `{cname}`")