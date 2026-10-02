"""
ESP32 Sensor Interface & Environmental Analysis Module
Connects to physical ESP32 over serial or runs in explicitly labelled simulation mode.
Reads Temperature (DHT22), Air Humidity (DHT22), and Soil Moisture (HW-080).
"""

import os
import json
import random
from typing import Dict, Any, List, Tuple, Optional
import serial
import serial.tools.list_ports

# Configuration defaults
DEFAULT_PORT = os.environ.get("ESP32_PORT", "COM3")
DEFAULT_BAUD = int(os.environ.get("ESP32_BAUD", 115200))
SERIAL_TIMEOUT = 2.0


def list_available_ports() -> List[str]:
    """
    Returns a list of available serial COM ports on the system.
    """
    try:
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    except Exception:
        return []


def connect_to_esp32(port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUD, timeout: float = SERIAL_TIMEOUT) -> serial.Serial:
    """
    Attempts to establish a serial connection to the ESP32 microcontroller.
    Raises ConnectionError if the port cannot be opened.
    """
    try:
        esp32 = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )
        return esp32
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to ESP32 on port {port}: {e}"
        )


def read_from_esp32(esp32: serial.Serial) -> Tuple[Dict[str, Any], str]:
    """
    Reads a single line of JSON from the open ESP32 serial connection.
    Expected format: {"temperature": 28.4, "humidity": 62.1, "soil_moisture": 68.0}
    Or error format: {"error": "DHT22_read_failed"}
    Returns (parsed_data_dict, raw_line_str).
    """
    try:
        raw_bytes = esp32.readline()
        if not raw_bytes:
            raise RuntimeError("No data received from ESP32 within timeout period.")

        line = raw_bytes.decode("utf-8", errors="replace").strip()
        if not line:
            raise RuntimeError("Empty response received from ESP32.")

        # Find first JSON object in line if mixed with bootloader logs
        start_idx = line.find("{")
        end_idx = line.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = line[start_idx : end_idx + 1]
        else:
            json_str = line

        data = json.loads(json_str)
        return data, line

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON received from ESP32: {e}")
    except Exception as e:
        raise ValueError(f"Failed reading ESP32 serial stream: {e}")


def get_simulated_sensor_data() -> Dict[str, Any]:
    """
    Returns clearly labelled simulated environmental readings for offline demonstration.
    """
    # Typical greenhouse conditions
    temp = round(26.0 + random.uniform(-2.5, 3.5), 1)
    humidity = round(58.0 + random.uniform(-5.0, 5.0), 1)
    moist = round(64.0 + random.uniform(-6.0, 6.0), 1)

    raw_json = json.dumps({
        "temperature": temp,
        "humidity": humidity,
        "soil_moisture": moist
    })

    return {
        "temperature_c": temp,
        "humidity_pct": humidity,
        "soil_moisture_pct": moist,
        "source": "Simulation",
        "status": "Simulation Active (Hardware not connected)",
        "raw_line": raw_json,
        "error": None
    }


def get_sensor_data(
    mode: str = "auto",
    port: Optional[str] = None,
    baudrate: int = DEFAULT_BAUD,
    timeout: float = SERIAL_TIMEOUT
) -> Dict[str, Any]:
    """
    High-level interface for fetching sensor readings.

    Parameters:
        mode: 'esp32', 'simulation', or 'auto' (tries esp32, falls back gracefully)
        port: COM port string (e.g. 'COM3', '/dev/ttyUSB0'). Defaults to DEFAULT_PORT.

    Returns:
        {
            "temperature_c": float or None,
            "humidity_pct": float or None,
            "soil_moisture_pct": float or None,
            "source": "ESP32" | "Simulation" | "Unavailable",
            "status": str,
            "raw_line": str or None,
            "error": Optional[str]
        }
    """
    selected_mode = mode.lower() if mode else "auto"

    if selected_mode == "simulation":
        return get_simulated_sensor_data()

    target_port = port or DEFAULT_PORT

    # Attempt physical ESP32 connection
    esp32_conn = None
    try:
        esp32_conn = connect_to_esp32(port=target_port, baudrate=baudrate, timeout=timeout)
        data, raw_line = read_from_esp32(esp32_conn)

        if "error" in data:
            hardware_err = data["error"]
            return {
                "temperature_c": None,
                "humidity_pct": None,
                "soil_moisture_pct": None,
                "source": "ESP32",
                "status": f"Hardware Sensor Warning: {hardware_err}",
                "raw_line": raw_line,
                "error": f"Sensor Error: {hardware_err}"
            }

        raw_temp = data.get("temperature")
        raw_hum = data.get("humidity")
        raw_moist = data.get("soil_moisture")

        temp_c = float(raw_temp) if raw_temp is not None else None
        hum_pct = float(raw_hum) if raw_hum is not None else None
        moist_pct = float(raw_moist) if raw_moist is not None else None

        return {
            "temperature_c": temp_c,
            "humidity_pct": hum_pct,
            "soil_moisture_pct": moist_pct,
            "source": "ESP32",
            "status": f"Connected to ESP32 on {target_port} ({baudrate} baud)",
            "raw_line": raw_line,
            "error": None
        }

    except Exception as e:
        error_msg = f"ESP32 on {target_port} unavailable: {e}"

        if selected_mode == "auto":
            # Gracefully provide simulated data with explicit warning
            sim_data = get_simulated_sensor_data()
            sim_data["status"] = f"Simulation (Hardware unavailable on {target_port})"
            sim_data["error"] = str(e)
            return sim_data

        # Explicit 'esp32' mode requested but failed
        return {
            "temperature_c": None,
            "humidity_pct": None,
            "soil_moisture_pct": None,
            "source": "Unavailable",
            "status": f"Disconnected ({target_port})",
            "raw_line": None,
            "error": error_msg
        }

    finally:
        if esp32_conn and esp32_conn.is_open:
            try:
                esp32_conn.close()
            except Exception:
                pass


def validate_sensor_data(
    temperature: Optional[float],
    soil_moisture: Optional[float],
    humidity: Optional[float] = None
) -> List[str]:
    """
    Validates physical sensor values within sensible biological and hardware ranges.
    """
    errors = []

    # Temperature validation (-10°C to 60°C)
    if temperature is None:
        errors.append("Temperature reading unavailable.")
    elif not -10.0 <= temperature <= 60.0:
        errors.append("Temperature reading is outside expected sensor range (-10°C to 60°C).")

    # Air Humidity validation (0% to 100%)
    if humidity is not None and not 0.0 <= humidity <= 100.0:
        errors.append("Air humidity percentage is outside valid range (0% to 100%).")

    # Soil moisture validation (0% to 100%)
    if soil_moisture is None:
        errors.append("Soil moisture reading unavailable.")
    elif not 0.0 <= soil_moisture <= 100.0:
        errors.append("Soil moisture percentage is outside valid range (0% to 100%).")

    return errors


def analyze_environment(
    temperature: Optional[float],
    soil_moisture: Optional[float],
    humidity: Optional[float] = None
) -> Tuple[List[str], str]:
    """
    Provides contextual environmental observations based on sensor measurements.

    Ranges:
    - Temperature: <15 (low), 15-30 (moderate), >30 (high)
    - Air Humidity: <40 (low), 40-75 (moderate), >75 (high)
    - Soil Moisture: <30 (low), 30-70 (moderate), >70 (high)
    """
    observations = []

    if temperature is None and soil_moisture is None and humidity is None:
        return (
            ["Environmental sensor data is currently unavailable."],
            "No sensor readings recorded for this analysis session."
        )

    # Temperature contextual assessment
    if temperature is not None:
        if temperature < 15.0:
            observations.append("Measured temperature is relatively low (< 15°C).")
        elif temperature <= 30.0:
            observations.append("Measured temperature is within a moderate range (15°C–30°C).")
        else:
            observations.append("Measured temperature is relatively high (> 30°C).")
    else:
        observations.append("Temperature sensor data is unavailable.")

    # Air Humidity contextual assessment (DHT22)
    if humidity is not None:
        if humidity < 40.0:
            observations.append("Air humidity is relatively low (< 40%). Dry atmospheric conditions.")
        elif humidity <= 75.0:
            observations.append("Air humidity is optimal for general crop canopy development (40%–75%).")
        else:
            observations.append("High air humidity (> 75%). Humid microclimates can favor fungal spore germination.")

    # Soil moisture contextual assessment (HW-080)
    if soil_moisture is not None:
        if soil_moisture < 30.0:
            observations.append("Measured soil moisture is relatively low (< 30%). Dry soil conditions.")
        elif soil_moisture <= 70.0:
            observations.append("Measured soil moisture is within a moderate range (30%–70%).")
        else:
            observations.append("Measured soil moisture is relatively high (> 70%). Soil is saturated.")
    else:
        observations.append("Soil moisture sensor data is unavailable.")

    overall = (
        "The DHT22 and HW-080 physical sensor readings provide microclimate context for the image-based "
        "disease prediction. They should not be interpreted as direct biological proof of disease."
    )

    return observations, overall