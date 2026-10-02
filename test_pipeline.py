"""
End-to-End Automated Test Suite for Crop Disease Detection & Environmental Monitoring System
Validates all 14 system verification points specified in Phase 9:
 1. Application imports and module integrity
 2. Model loading and inference with real checkpoint
 3. Valid serial JSON parsing
 4. Missing fields and malformed JSON
 5. Invalid numeric values, NaN, infinity, and out-of-range readings
 6. Serial timeout and disconnection handling
 7. Stale-reading detection
 8. Reconnection and resource cleanup
 9. Simulation Mode behavior
10. Live Sensor Mode never invoking simulation as a fallback
11. Correct mode/source attribution in history and exports
12. Database persistence and preservation of existing records
13. Switching modes across session state logic
14. Behavior when the ESP32 is disconnected
"""

import sys
import time
import math
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model import predict_leaf, CLASS_NAMES, ensure_model_file, get_model
from sensors import (
    get_sensor_data,
    validate_sensor_data,
    analyze_environment,
    parse_sensor_json,
    SerialManager,
    read_from_esp32_sync,
    DEFAULT_BAUD,
    TEMP_MIN,
    TEMP_MAX,
    HUM_MIN,
    HUM_MAX,
    SOIL_MIN,
    SOIL_MAX
)
from database import (
    init_database,
    save_analysis,
    get_analysis_history,
    get_analysis_by_id,
    delete_analysis,
    get_statistics
)
from utils import (
    generate_sample_leaf_image,
    generate_text_report,
    get_confidence_interpretation
)


def run_all_tests():
    print("=" * 70)
    print("CROP DISEASE DETECTION & IOT HARDWARE: AUTOMATED VERIFICATION SUITE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TEST 1: Application Imports & Module Integrity
    # -------------------------------------------------------------------------
    print("\n[TEST 1/14] Testing Application Imports & Module Integrity...")
    for mod in ["torch", "torchvision", "streamlit", "serial", "PIL", "model", "sensors", "database", "utils"]:
        __import__(mod)
    print("✓ All core modules (PyTorch, Torchvision, Streamlit, PySerial, Pillow) imported cleanly.")

    # -------------------------------------------------------------------------
    # TEST 2: Model Loading & Inference
    # -------------------------------------------------------------------------
    print("\n[TEST 2/14] Testing Model Loading & Inference with Checkpoint...")
    model_path = ensure_model_file()
    assert model_path.exists() and model_path.stat().st_size > 0, "Model checkpoint missing or empty!"
    assert len(CLASS_NAMES) == 38, f"Expected 38 classes, got {len(CLASS_NAMES)}"

    test_img = generate_sample_leaf_image(plant="Tomato", healthy=False)
    pred_res = predict_leaf(test_img, top_k=5)
    assert "predicted_class" in pred_res
    assert "plant" in pred_res
    assert "condition" in pred_res
    assert "confidence" in pred_res
    assert len(pred_res["top_predictions"]) == 5
    assert 0.0 <= pred_res["confidence"] <= 1.0
    print(f"✓ Model inference succeeded: {pred_res['predicted_class']} ({pred_res['confidence'] * 100:.2f}%)")

    # -------------------------------------------------------------------------
    # TEST 3: Valid Serial JSON Parsing
    # -------------------------------------------------------------------------
    print("\n[TEST 3/14] Testing Valid Serial JSON Parsing...")
    valid_json = '{"temperature": 25.4, "humidity": 60.1, "soil_moisture": 45.2}'
    data, err = parse_sensor_json(valid_json)
    assert data is not None, f"Failed parsing valid JSON: {err}"
    assert data["temperature_c"] == 25.4
    assert data["humidity_pct"] == 60.1
    assert data["soil_moisture_pct"] == 45.2

    # Noise / Bootloader prefix test
    noisy_line = 'rst:0x1 (POWERON_RESET),boot:0x13... {"temperature": 28.1, "humidity": 65.0, "soil_moisture": 50.0}\r\n'
    data_noisy, _ = parse_sensor_json(noisy_line)
    assert data_noisy is not None
    assert data_noisy["temperature_c"] == 28.1
    assert data_noisy["humidity_pct"] == 65.0
    assert data_noisy["soil_moisture_pct"] == 50.0
    print("✓ Valid JSON and noisy bootloader JSON parsed accurately.")

    # -------------------------------------------------------------------------
    # TEST 4: Missing Fields & Malformed JSON
    # -------------------------------------------------------------------------
    print("\n[TEST 4/14] Testing Missing Fields & Malformed JSON...")
    # Malformed syntax
    data_bad, err_bad = parse_sensor_json('{"temperature": 25.4, "humidity":')
    assert data_bad is None
    assert "Malformed" in err_bad or "incomplete" in err_bad

    # Empty and non-dict inputs
    assert parse_sensor_json("")[0] is None
    assert parse_sensor_json("[1, 2, 3]")[0] is None

    # Error packet from firmware (DHT22 failed, Soil Moisture succeeded)
    err_pkt = '{"error": "DHT22_read_failed", "soil_moisture": 52.3}'
    data_err, note_err = parse_sensor_json(err_pkt)
    assert data_err is not None
    assert data_err["temperature_c"] is None
    assert data_err["humidity_pct"] is None
    assert data_err["soil_moisture_pct"] == 52.3
    assert "DHT22_read_failed" in data_err["firmware_error"]
    print("✓ Malformed lines and partial error packets handled safely.")

    # -------------------------------------------------------------------------
    # TEST 5: Invalid Numeric Values, NaN, Infinity & Out-of-Range Readings
    # -------------------------------------------------------------------------
    print("\n[TEST 5/14] Testing Invalid Numerics, NaN, Infinity & Out-of-Range...")
    # Non-numeric string
    d, _ = parse_sensor_json('{"temperature": "hot", "humidity": 50.0, "soil_moisture": 50.0}')
    assert d["temperature_c"] is None
    assert d["humidity_pct"] == 50.0

    # Boolean value (must not be parsed as 1.0)
    d, _ = parse_sensor_json('{"temperature": true, "humidity": 50.0, "soil_moisture": 50.0}')
    assert d["temperature_c"] is None

    # NaN / Infinity strings
    d, _ = parse_sensor_json('{"temperature": "NaN", "humidity": "Infinity", "soil_moisture": 50.0}')
    assert d["temperature_c"] is None
    assert d["humidity_pct"] is None

    # Out of physical bounds (must NOT be clamped into fake valid numbers!)
    d, _ = parse_sensor_json('{"temperature": 150.0, "humidity": -20.0, "soil_moisture": 180.0}')
    assert d is None  # all fields invalid -> total rejection

    d2, _ = parse_sensor_json('{"temperature": 25.0, "humidity": 150.0, "soil_moisture": 50.0}')
    assert d2["temperature_c"] == 25.0
    assert d2["humidity_pct"] is None  # out-of-range humidity rejected
    print("✓ Out-of-range, NaN, Infinity, and boolean inputs rejected without silent clamping.")

    # -------------------------------------------------------------------------
    # TEST 6: Serial Timeout & Disconnection Handling
    # -------------------------------------------------------------------------
    print("\n[TEST 6/14] Testing Serial Timeout & Disconnection Handling...")
    res = read_from_esp32_sync(port="COM999", timeout=0.1)
    assert res["source"] == "Unavailable"
    assert res["temperature_c"] is None
    assert res["humidity_pct"] is None
    assert res["connection_state"] == "Disconnected"
    assert "unavailable" in res["error"].lower() or "failed" in res["error"].lower()
    print(f"✓ Disconnected port COM999 handled cleanly without crash: {res['status']}")

    # -------------------------------------------------------------------------
    # TEST 7: Stale-Reading Detection
    # -------------------------------------------------------------------------
    print("\n[TEST 7/14] Testing Stale-Reading Detection...")
    mgr = SerialManager.get_instance()
    # Inject an artificially old reading
    with mgr.data_lock:
        mgr.port = "COM_TEST"
        mgr.baudrate = 115200
        mgr.connection_state = "Connected"
        mgr.last_valid_reading = {"temperature_c": 24.5, "humidity_pct": 55.0, "soil_moisture_pct": 60.0}
        mgr.last_valid_time = time.time() - 15.0  # 15 seconds ago

    stale_res = mgr.get_reading(stale_timeout=5.0)
    assert stale_res["connection_state"] == "Stale"
    assert stale_res["source"] == "Unavailable"
    assert stale_res["temperature_c"] is None
    assert stale_res["data_age"] >= 15.0
    assert "stale" in stale_res["error"].lower()

    # Reset manager state
    with mgr.data_lock:
        mgr.port = None
        mgr.connection_state = "Disconnected"
        mgr.last_valid_reading = None
        mgr.last_valid_time = None
    print(f"✓ Stale reading correctly identified (> 5.0s age). Data marked Unavailable.")

    # -------------------------------------------------------------------------
    # TEST 8: Reconnection & Resource Cleanup
    # -------------------------------------------------------------------------
    print("\n[TEST 8/14] Testing Reconnection & Resource Cleanup...")
    # Disconnecting an already disconnected manager should not throw
    mgr.disconnect()
    assert mgr.connection_state == "Disconnected"
    assert mgr.serial_conn is None

    # Connecting to invalid port cleans up resources safely
    success, msg = mgr.connect("COM_NONEXISTENT", 115200)
    assert not success
    assert mgr.connection_state == "Error"
    mgr.disconnect()
    assert mgr.connection_state == "Disconnected"
    print("✓ Resource cleanup and idempotent disconnection verified.")

    # -------------------------------------------------------------------------
    # TEST 9: Simulation Mode Behavior
    # -------------------------------------------------------------------------
    print("\n[TEST 9/14] Testing Simulation Mode Behavior...")
    sim_data = get_sensor_data(mode="simulation")
    assert sim_data["source"] == "Simulation"
    assert sim_data["temperature_c"] is not None
    assert sim_data["humidity_pct"] is not None
    assert sim_data["soil_moisture_pct"] is not None
    assert "simulation" in sim_data["status"].lower()
    print(f"✓ Simulation Mode generated synthetic data: {sim_data['temperature_c']}°C, {sim_data['humidity_pct']}% RH, {sim_data['soil_moisture_pct']}% (Source: {sim_data['source']})")

    # -------------------------------------------------------------------------
    # TEST 10: Live Sensor Mode Never Silently Falls Back to Simulation
    # -------------------------------------------------------------------------
    print("\n[TEST 10/14] Testing Zero Silent Simulation Fallback in Live Mode...")
    live_data = get_sensor_data(mode="live", port="COM999", use_sync=True)
    assert live_data["source"] != "Simulation", "CRITICAL DEFECT: Live mode silently substituted simulation data!"
    assert live_data["source"] == "Unavailable"
    assert live_data["temperature_c"] is None
    assert live_data["humidity_pct"] is None
    assert live_data["soil_moisture_pct"] is None
    print("✓ Confirmed: Live Sensor Mode NEVER substitutes simulated data when hardware fails.")

    # -------------------------------------------------------------------------
    # TEST 11: Correct Mode/Source Attribution in History and Exports
    # -------------------------------------------------------------------------
    print("\n[TEST 11/14] Testing Mode & Source Attribution in History & Reports...")
    sim_report_entry = {
        "plant": "Tomato",
        "predicted_condition": "healthy",
        "predicted_class": "Tomato___healthy",
        "confidence": 0.95,
        "temperature_c": 25.0,
        "humidity_pct": 60.0,
        "soil_moisture_pct": 55.0,
        "sensor_source": "Simulation"
    }
    rep_sim = generate_text_report(sim_report_entry)
    assert "SIMULATED" in rep_sim
    assert "Offline Demonstration" in rep_sim

    esp_report_entry = {
        "plant": "Tomato",
        "predicted_condition": "healthy",
        "predicted_class": "Tomato___healthy",
        "confidence": 0.95,
        "temperature_c": 24.2,
        "humidity_pct": 58.1,
        "soil_moisture_pct": 50.0,
        "sensor_source": "ESP32"
    }
    rep_esp = generate_text_report(esp_report_entry)
    assert "ESP32 Physical Microcontroller" in rep_esp
    assert "SIMULATED" not in rep_esp
    print("✓ Text reports accurately differentiate physical ESP32 telemetry from Simulation.")

    # -------------------------------------------------------------------------
    # TEST 12: Database Persistence & Preservation of Existing Records
    # -------------------------------------------------------------------------
    print("\n[TEST 12/14] Testing Database Persistence & Preservation...")
    init_database()
    initial_history = get_analysis_history(limit=100)
    initial_count = len(initial_history)

    # Insert an ESP32 record
    esp_id = save_analysis({
        "image_name": "test_esp32_leaf.jpg",
        "plant": "Pepper, bell",
        "predicted_condition": "Bacterial spot",
        "predicted_class": "Pepper,_bell___Bacterial_spot",
        "confidence": 0.88,
        "temperature_c": 27.5,
        "humidity_pct": 68.0,
        "soil_moisture_pct": 42.0,
        "sensor_source": "ESP32",
        "top_predictions": [{"plant": "Pepper, bell", "condition": "Bacterial spot", "confidence": 0.88}],
        "environmental_observations": ["High humidity microclimate"],
        "environmental_summary": "Favorable fungal/bacterial conditions"
    })

    # Insert a Simulation record
    sim_id = save_analysis({
        "image_name": "test_sim_leaf.jpg",
        "plant": "Tomato",
        "predicted_condition": "Early blight",
        "predicted_class": "Tomato___Early_blight",
        "confidence": 0.79,
        "temperature_c": 26.0,
        "humidity_pct": 58.0,
        "soil_moisture_pct": 65.0,
        "sensor_source": "Simulation",
        "top_predictions": [{"plant": "Tomato", "condition": "Early blight", "confidence": 0.79}],
        "environmental_observations": ["Optimal soil moisture"],
        "environmental_summary": "Demonstration simulation summary"
    })

    new_history = get_analysis_history(limit=100)
    assert len(new_history) == initial_count + 2, f"Expected {initial_count + 2} records, found {len(new_history)}"

    fetched_esp = get_analysis_by_id(esp_id)
    assert fetched_esp["sensor_source"] == "ESP32"
    assert fetched_esp["temperature_c"] == 27.5

    fetched_sim = get_analysis_by_id(sim_id)
    assert fetched_sim["sensor_source"] == "Simulation"
    assert fetched_sim["temperature_c"] == 26.0

    # Clean up test rows
    delete_analysis(esp_id)
    delete_analysis(sim_id)
    cleaned_history = get_analysis_history(limit=100)
    assert len(cleaned_history) == initial_count, "Database cleanup failed to restore exact original count!"
    print(f"✓ Database persistence verified: Preserved existing {initial_count} records, saved & verified ESP32 + Simulation records.")

    # -------------------------------------------------------------------------
    # TEST 13: Switching Modes Across Session State Logic
    # -------------------------------------------------------------------------
    print("\n[TEST 13/14] Testing Mode Switching Logic...")
    # Simulate switching from Simulation to Live
    active_mode = "Simulation Mode"
    data1 = get_sensor_data(mode=active_mode)
    assert data1["source"] == "Simulation"

    # Switch to Live Mode
    active_mode = "Live Sensor Mode"
    data2 = get_sensor_data(mode="live", port="COM999", use_sync=True)
    assert data2["source"] == "Unavailable"
    assert data2["source"] != "Simulation"

    # Switch back to Simulation Mode
    active_mode = "Simulation Mode"
    data3 = get_sensor_data(mode="simulation")
    assert data3["source"] == "Simulation"
    print("✓ Mode switching maintains strict boundary between Simulation and Live Sensor states.")

    # -------------------------------------------------------------------------
    # TEST 14: Behavior When ESP32 is Disconnected
    # -------------------------------------------------------------------------
    print("\n[TEST 14/14] Testing Workflow When ESP32 is Disconnected...")
    # 1. Hardware is disconnected
    disc_sensor = get_sensor_data(mode="live", port="COM999", use_sync=True)
    assert disc_sensor["source"] == "Unavailable"

    # 2. Image inference must still succeed independently
    leaf_img = generate_sample_leaf_image(plant="Tomato", healthy=True)
    pred = predict_leaf(leaf_img)
    assert pred["predicted_class"] is not None

    # 3. Environmental analysis handles None gracefully without crash
    obs, summary = analyze_environment(disc_sensor["temperature_c"], disc_sensor["soil_moisture_pct"], disc_sensor["humidity_pct"])
    assert "unavailable" in obs[0].lower()
    assert "visual symptoms" in summary.lower()

    # 4. Record can still be persisted with source='Unavailable'
    disc_id = save_analysis({
        "image_name": "disconnected_leaf_test.jpg",
        "plant": pred["plant"],
        "predicted_condition": pred["condition"],
        "predicted_class": pred["predicted_class"],
        "confidence": pred["confidence"],
        "temperature_c": None,
        "humidity_pct": None,
        "soil_moisture_pct": None,
        "sensor_source": "Unavailable",
        "top_predictions": pred["top_predictions"],
        "environmental_observations": obs,
        "environmental_summary": summary
    })
    assert disc_id > 0
    delete_analysis(disc_id)
    print("✓ Image classification and database logging continue to function honestly when ESP32 is disconnected.")

    print("\n" + "=" * 70)
    print("ALL 14 AUTOMATED TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
