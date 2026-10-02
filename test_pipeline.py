"""
End-to-End Test Suite for Crop Disease Detection & Environmental Monitoring System
Validates Model inference, Sensor communication, SQLite database, and Utility components.
"""

import sys
from pathlib import Path
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from model import predict_leaf, CLASS_NAMES, ensure_model_file
from sensors import get_sensor_data, validate_sensor_data, analyze_environment
from database import (
    init_database,
    save_analysis,
    get_analysis_history,
    get_analysis_by_id,
    delete_analysis,
    get_statistics,
    clear_history
)
from utils import generate_sample_leaf_image, generate_text_report, get_confidence_interpretation


def run_tests():
    print("=" * 60)
    print("STARTING COMPLETE SYSTEM PIPELINE TESTS")
    print("=" * 60)

    # 1. Test database initialization
    print("\n[1/10] Testing SQLite Database Initialization...")
    init_database()
    assert (PROJECT_ROOT / "crop_disease.db").exists(), "Database file crop_disease.db was not created!"
    print("✓ SQLite database initialized successfully.")

    # 2. Test model checkpoint availability and loading
    print("\n[2/10] Testing Model Checkpoint & Inference...")
    model_path = ensure_model_file()
    assert model_path.exists() and model_path.stat().st_size > 0, "Model checkpoint is missing or empty!"
    assert len(CLASS_NAMES) == 38, f"Expected 38 classes, found {len(CLASS_NAMES)}"

    # Generate synthetic leaf
    test_leaf = generate_sample_leaf_image(plant="Tomato", healthy=False)
    pred_res = predict_leaf(test_leaf, top_k=5)

    assert "predicted_class" in pred_res
    assert "plant" in pred_res
    assert "condition" in pred_res
    assert "confidence" in pred_res
    assert len(pred_res["top_predictions"]) == 5
    print(f"✓ Model inference passed! Class: {pred_res['predicted_class']}, Confidence: {pred_res['confidence']:.4f}")
    print(f"✓ Top 5 predictions: {[p['readable'] for p in pred_res['top_predictions']]}")

    # 3. Test sensor module in simulation mode
    print("\n[3/10] Testing Sensor Module in Simulation Mode...")
    sim_data = get_sensor_data(mode="simulation")
    assert sim_data["source"] == "Simulation"
    assert sim_data["temperature_c"] is not None
    assert sim_data["humidity_pct"] is not None
    assert sim_data["soil_moisture_pct"] is not None
    print(f"✓ Simulation data: Temp={sim_data['temperature_c']}°C, Air Humidity={sim_data['humidity_pct']}%, Soil Moisture={sim_data['soil_moisture_pct']}%, Source={sim_data['source']}")

    # 4. Test sensor module in ESP32 disconnected mode (Graceful handling)
    print("\n[4/10] Testing ESP32 Disconnected Handling...")
    esp_data = get_sensor_data(mode="esp32", port="COM999")
    assert esp_data["source"] == "Unavailable"
    assert esp_data["temperature_c"] is None
    assert esp_data["humidity_pct"] is None
    assert esp_data["error"] is not None
    print(f"✓ Graceful disconnection handled without crash: {esp_data['status']}")

    # 5. Test sensor validation and environmental analysis
    print("\n[5/10] Testing Sensor Validation & Context Analysis...")
    v_errors = validate_sensor_data(sim_data["temperature_c"], sim_data["soil_moisture_pct"], sim_data["humidity_pct"])
    assert len(v_errors) == 0, f"Unexpected validation errors: {v_errors}"

    obs, summary = analyze_environment(sim_data["temperature_c"], sim_data["soil_moisture_pct"], sim_data["humidity_pct"])
    assert len(obs) >= 3, "Expected at least 3 environmental observations (Temp, Air Humidity, Soil Moisture)"
    assert "context" in summary.lower()
    print(f"✓ Environmental analysis passed: {obs}")

    # 6. Test saving analysis to SQLite
    print("\n[6/10] Testing SQLite Save Analysis...")
    analysis_record = {
        "image_name": "test_leaf_simulation.jpg",
        "plant": pred_res["plant"],
        "predicted_condition": pred_res["condition"],
        "predicted_class": pred_res["predicted_class"],
        "confidence": pred_res["confidence"],
        "temperature_c": sim_data["temperature_c"],
        "humidity_pct": sim_data["humidity_pct"],
        "soil_moisture_pct": sim_data["soil_moisture_pct"],
        "sensor_source": sim_data["source"],
        "top_predictions": pred_res["top_predictions"],
        "environmental_observations": obs,
        "environmental_summary": summary
    }
    inserted_id = save_analysis(analysis_record)
    assert inserted_id > 0, "Insert failed to return a valid ID"
    print(f"✓ Record saved successfully with ID: {inserted_id}")

    # 7. Test retrieving history from SQLite
    print("\n[7/10] Testing SQLite History Retrieval...")
    history = get_analysis_history(limit=10)
    assert len(history) > 0, "History is empty after insert"
    fetched = history[0]
    assert fetched["id"] == inserted_id
    assert fetched["plant"] == pred_res["plant"]
    assert fetched["humidity_pct"] == sim_data["humidity_pct"]
    assert len(fetched["top_predictions"]) == 5
    print(f"✓ History record #{fetched['id']} verified with {len(fetched['top_predictions'])} top predictions.")

    # 8. Test report generation
    print("\n[8/10] Testing Report Generation...")
    report_text = generate_text_report(fetched)
    assert "PLANT HEALTH & ENVIRONMENTAL MONITORING REPORT" in report_text
    assert "Air Humidity (DHT22)" in report_text
    assert pred_res["plant"] in report_text
    print("✓ Full text report generated successfully.")

    # 9. Test Statistics calculation
    print("\n[9/10] Testing Database Statistics...")
    stats = get_statistics()
    assert stats["total_analyses"] >= 1
    print(f"✓ Aggregate stats verified: Total={stats['total_analyses']}, Healthy={stats['healthy_count']}, Diseased={stats['diseased_count']}")

    # 10. Check for no remaining MongoDB references in codebase
    print("\n[10/10] Checking for Zero MongoDB Dependencies...")
    for py_file in ["app.py", "model.py", "database.py", "sensors.py", "utils.py"]:
        file_path = PROJECT_ROOT / py_file
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            assert "pymongo" not in content, f"Found pymongo in {py_file}!"
            assert "MongoClient" not in content, f"Found MongoClient in {py_file}!"
    print("✓ Confirmed: Zero MongoDB references in all application source files.")


    print("\n" + "=" * 60)
    print("ALL 10 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
