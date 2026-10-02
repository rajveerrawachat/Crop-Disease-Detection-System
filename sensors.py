"""
ESP32 Sensor Interface & Environmental Analysis Module
Provides robust USB serial communication for ESP32 with DHT22 (GPIO 4) and Soil Moisture (GPIO 34),
along with a thread-safe background SerialManager and an explicit Simulation Mode.

Operating Modes:
  1. Simulation Mode  - Explicitly generates synthetic measurements labelled as 'Simulation'.
  2. Live Sensor Mode - Reads real, validated packets from ESP32. Never falls back to simulation.
"""

import os
import json
import time
import math
import random
import threading
from typing import Dict, Any, List, Tuple, Optional
import serial
import serial.tools.list_ports

# Configuration defaults
DEFAULT_PORT = os.environ.get("ESP32_PORT", "COM3")
DEFAULT_BAUD = int(os.environ.get("ESP32_BAUD", 115200))
DEFAULT_STALE_TIMEOUT = 8.0  # seconds before reading is considered stale
DEFAULT_READ_TIMEOUT = 1.5   # serial read timeout in seconds

# Physical sensor validation boundaries
TEMP_MIN, TEMP_MAX = -10.0, 60.0       # DHT22 sensible greenhouse range (°C)
HUM_MIN, HUM_MAX = 0.0, 100.0          # DHT22 relative humidity (%)
SOIL_MIN, SOIL_MAX = 0.0, 100.0        # HW-080 calibrated moisture (%)


def list_available_ports() -> List[str]:
    """
    Returns a list of available serial COM ports on the system.
    """
    try:
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    except Exception:
        return []


def parse_sensor_json(raw_line: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parses and validates a single line of JSON from the ESP32.
    Expected normal payload: {"temperature": 25.4, "humidity": 60.1, "soil_moisture": 45.2}
    Expected error payload:  {"error": "DHT22_read_failed", "soil_moisture": 45.2}

    Returns:
        (validated_dict, error_message)
        If parsing succeeds, validated_dict contains floats (or None for failed sensors).
        If entirely invalid or unparseable, returns (None, error_message).
    """
    if not raw_line or not isinstance(raw_line, str):
        return None, "Empty or non-string serial data received."

    clean_line = raw_line.strip()
    if not clean_line:
        return None, "Empty line received from serial stream."

    # Extract JSON substring if surrounded by ESP32 bootloader logs or garbage
    start_idx = clean_line.find("{")
    end_idx = clean_line.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None, f"Malformed or incomplete JSON object: '{clean_line}'"

    json_str = clean_line[start_idx : end_idx + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Malformed JSON: {e}"

    if not isinstance(data, dict):
        return None, "Received JSON payload is not a dictionary."

    # Check for firmware-level sensor read error
    firmware_error = data.get("error")

    # Helper function to validate numeric fields
    def validate_numeric(val: Any, min_val: float, max_val: float, field_name: str) -> Tuple[Optional[float], Optional[str]]:
        if val is None:
            return None, None
        try:
            # Reject bools which are instances of int in Python
            if isinstance(val, bool):
                return None, f"Field '{field_name}' must be numeric, received boolean."
            f_val = float(val)
        except (ValueError, TypeError):
            return None, f"Field '{field_name}' contains non-numeric value: {val}"

        if math.isnan(f_val) or math.isinf(f_val):
            return None, f"Field '{field_name}' contains non-finite number (NaN/Inf)."

        if not (min_val <= f_val <= max_val):
            return None, f"Field '{field_name}' value {f_val} is out of expected physical bounds [{min_val}, {max_val}]."

        return round(f_val, 1), None

    temp_val, temp_err = validate_numeric(data.get("temperature"), TEMP_MIN, TEMP_MAX, "temperature")
    hum_val, hum_err = validate_numeric(data.get("humidity"), HUM_MIN, HUM_MAX, "humidity")
    soil_val, soil_err = validate_numeric(data.get("soil_moisture"), SOIL_MIN, SOIL_MAX, "soil_moisture")

    # Collect validation warnings/errors
    errors = [e for e in [temp_err, hum_err, soil_err] if e is not None]
    if firmware_error:
        errors.insert(0, f"Hardware sensor warning: {firmware_error}")

    # If all sensor readings are None and an error exists, report total failure
    if temp_val is None and hum_val is None and soil_val is None:
        err_msg = "; ".join(errors) if errors else "Payload contains no valid sensor measurements."
        return None, err_msg

    validated = {
        "temperature_c": temp_val,
        "humidity_pct": hum_val,
        "soil_moisture_pct": soil_val,
        "firmware_error": firmware_error,
        "validation_notes": errors
    }
    return validated, ("; ".join(errors) if errors else None)


def get_simulated_sensor_data() -> Dict[str, Any]:
    """
    Returns clearly labelled simulated environmental readings for offline demonstration.
    """
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
        "data_age": 0.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_line": raw_json,
        "error": None
    }


class SerialManager:
    """
    Thread-safe Singleton Manager for the ESP32 USB Serial connection.
    Runs a background reader thread so the COM port is kept open continuously across
    Streamlit script reruns without toggling DTR/RTS and resetting the ESP32.
    """
    _instance: Optional["SerialManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.port: Optional[str] = None
        self.baudrate: int = DEFAULT_BAUD
        self.serial_conn: Optional[serial.Serial] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Thread-safe telemetry state
        self.data_lock = threading.Lock()
        self.last_valid_reading: Optional[Dict[str, Any]] = None
        self.last_valid_time: Optional[float] = None
        self.last_raw_line: Optional[str] = None
        self.connection_state: str = "Disconnected"  # "Disconnected", "Connecting", "Connected", "Error", "Stale"
        self.error_message: Optional[str] = None
        self.recent_logs: List[Dict[str, str]] = []

    @classmethod
    def get_instance(cls) -> "SerialManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def connect(self, port: str, baudrate: int = DEFAULT_BAUD) -> Tuple[bool, str]:
        """
        Connects to the specified COM port and starts the background reader.
        If already connected to the same port and baudrate, reuses connection.
        """
        with self.data_lock:
            if (
                self.serial_conn is not None
                and self.serial_conn.is_open
                and self.port == port
                and self.baudrate == baudrate
                and self.reader_thread is not None
                and self.reader_thread.is_alive()
            ):
                return True, f"Already connected to {port} @ {baudrate} baud."

        # Disconnect any existing connection first
        self.disconnect()

        try:
            conn = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=DEFAULT_READ_TIMEOUT
            )
            # Flush existing buffers
            conn.reset_input_buffer()
        except serial.SerialException as e:
            with self.data_lock:
                self.connection_state = "Error"
                self.error_message = f"Failed to open port {port}: {e}"
                self._append_log(f"ERR: {self.error_message}", "err")
            return False, self.error_message
        except Exception as e:
            with self.data_lock:
                self.connection_state = "Error"
                self.error_message = f"Unexpected error opening port {port}: {e}"
                self._append_log(f"ERR: {self.error_message}", "err")
            return False, self.error_message

        with self.data_lock:
            self.port = port
            self.baudrate = baudrate
            self.serial_conn = conn
            self.stop_event.clear()
            self.connection_state = "Connecting"
            self.error_message = None
            self.last_valid_reading = None
            self.last_valid_time = None
            self._append_log(f"INFO: Opened serial port {port} at {baudrate} baud. Awaiting initial telemetry...", "info")

        # Start background reader thread
        self.reader_thread = threading.Thread(
            target=self._reader_worker,
            name="ESP32-SerialReader",
            daemon=True
        )
        self.reader_thread.start()
        return True, f"Connected to {port}. Waiting for telemetry packets."

    def disconnect(self):
        """
        Stops the reader thread and safely closes the serial connection.
        """
        self.stop_event.set()

        conn_to_close = None
        with self.data_lock:
            if self.serial_conn is not None:
                conn_to_close = self.serial_conn
                self.serial_conn = None
            self.connection_state = "Disconnected"
            self.error_message = None
            self._append_log("INFO: Serial connection closed.", "info")

        if conn_to_close is not None:
            try:
                conn_to_close.close()
            except Exception:
                pass

        if self.reader_thread is not None and self.reader_thread.is_alive():
            # Wait up to 1 second for thread to terminate
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None

    def _append_log(self, text: str, log_type: str = "out"):
        ts = time.strftime("%H:%M:%S")
        self.recent_logs.append({"timestamp": ts, "text": text, "type": log_type})
        if len(self.recent_logs) > 50:
            self.recent_logs.pop(0)

    def _reader_worker(self):
        """
        Continuous loop reading lines from ESP32 USB serial.
        """
        while not self.stop_event.is_set():
            conn = self.serial_conn
            if conn is None or not conn.is_open:
                break

            try:
                raw_bytes = conn.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Record log
                with self.data_lock:
                    self.last_raw_line = line
                    self._append_log(f"RX: {line}", "out")

                # Parse and validate packet
                validated_data, parse_err = parse_sensor_json(line)

                with self.data_lock:
                    if validated_data is not None:
                        self.last_valid_reading = validated_data
                        self.last_valid_time = time.time()
                        self.connection_state = "Connected"
                        self.error_message = parse_err  # None or warning notes
                    else:
                        # Packet was malformed or error
                        if parse_err:
                            self._append_log(f"PARSE_WARN: {parse_err}", "err")

            except serial.SerialException as e:
                with self.data_lock:
                    self.connection_state = "Error"
                    self.error_message = f"Serial port read error: {e}"
                    self._append_log(f"ERR: {self.error_message}", "err")
                break
            except Exception as e:
                with self.data_lock:
                    self.error_message = f"Reader loop error: {e}"
                    self._append_log(f"ERR: {self.error_message}", "err")
                time.sleep(0.1)

    def get_reading(self, stale_timeout: float = DEFAULT_STALE_TIMEOUT) -> Dict[str, Any]:
        """
        Retrieves the latest validated telemetry snapshot.
        Evaluates connection state and staleness. NEVER falls back to simulation.
        """
        with self.data_lock:
            state = self.connection_state
            last_reading = self.last_valid_reading
            last_time = self.last_valid_time
            port = self.port
            baud = self.baudrate
            raw_line = self.last_raw_line
            err = self.error_message

        now = time.time()

        if state == "Disconnected" or port is None:
            return {
                "temperature_c": None,
                "humidity_pct": None,
                "soil_moisture_pct": None,
                "source": "Unavailable",
                "status": "Hardware Disconnected",
                "connection_state": "Disconnected",
                "port": port,
                "baudrate": baud,
                "data_age": None,
                "timestamp": None,
                "raw_line": None,
                "error": "ESP32 hardware is not connected. Select COM port and connect."
            }

        if state == "Connecting":
            return {
                "temperature_c": None,
                "humidity_pct": None,
                "soil_moisture_pct": None,
                "source": "Unavailable",
                "status": f"Connecting to {port} (Awaiting first valid packet)",
                "connection_state": "Connecting",
                "port": port,
                "baudrate": baud,
                "data_age": None,
                "timestamp": None,
                "raw_line": raw_line,
                "error": "Port is open, waiting for incoming telemetry packet..."
            }

        if state == "Error":
            return {
                "temperature_c": None,
                "humidity_pct": None,
                "soil_moisture_pct": None,
                "source": "Unavailable",
                "status": f"Hardware Error on {port}",
                "connection_state": "Error",
                "port": port,
                "baudrate": baud,
                "data_age": None,
                "timestamp": None,
                "raw_line": raw_line,
                "error": err or "Serial communication error occurred."
            }

        # Check staleness of valid reading
        if last_reading is not None and last_time is not None:
            age = now - last_time
            if age > stale_timeout:
                return {
                    "temperature_c": None,
                    "humidity_pct": None,
                    "soil_moisture_pct": None,
                    "source": "Unavailable",
                    "status": f"Stale Data (Last packet received {age:.1f}s ago)",
                    "connection_state": "Stale",
                    "port": port,
                    "baudrate": baud,
                    "data_age": round(age, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_time)),
                    "raw_line": raw_line,
                    "error": f"Sensor readings are stale (> {stale_timeout}s without fresh packet)."
                }

            # Valid, fresh data from physical ESP32
            return {
                "temperature_c": last_reading.get("temperature_c"),
                "humidity_pct": last_reading.get("humidity_pct"),
                "soil_moisture_pct": last_reading.get("soil_moisture_pct"),
                "source": "ESP32",
                "status": f"Connected to ESP32 on {port} ({baud} baud)",
                "connection_state": "Connected",
                "port": port,
                "baudrate": baud,
                "data_age": round(age, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_time)),
                "raw_line": raw_line,
                "error": err
            }

        # Fallback if connected but no packet has arrived
        return {
            "temperature_c": None,
            "humidity_pct": None,
            "soil_moisture_pct": None,
            "source": "Unavailable",
            "status": f"Connected to {port} (No valid packet received yet)",
            "connection_state": "Connecting",
            "port": port,
            "baudrate": baud,
            "data_age": None,
            "timestamp": None,
            "raw_line": raw_line,
            "error": "Port is open, awaiting valid sensor telemetry..."
        }

    def get_logs(self) -> List[Dict[str, str]]:
        with self.data_lock:
            return list(self.recent_logs)

    def clear_logs(self):
        with self.data_lock:
            self.recent_logs = []


def read_from_esp32_sync(
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUD,
    timeout: float = DEFAULT_READ_TIMEOUT,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """
    Synchronous direct read function for unit tests and scripts.
    Attempts to read one valid JSON line from the ESP32.
    NEVER falls back to simulated data.
    """
    esp32_conn = None
    try:
        esp32_conn = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        last_raw_line = None
        for _ in range(max_attempts):
            raw_bytes = esp32_conn.readline()
            if not raw_bytes:
                continue
            line = raw_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            last_raw_line = line
            validated, err = parse_sensor_json(line)
            if validated is not None:
                return {
                    "temperature_c": validated.get("temperature_c"),
                    "humidity_pct": validated.get("humidity_pct"),
                    "soil_moisture_pct": validated.get("soil_moisture_pct"),
                    "source": "ESP32",
                    "status": f"Connected to ESP32 on {port} ({baudrate} baud)",
                    "connection_state": "Connected",
                    "port": port,
                    "baudrate": baudrate,
                    "data_age": 0.0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_line": line,
                    "error": err
                }

        return {
            "temperature_c": None,
            "humidity_pct": None,
            "soil_moisture_pct": None,
            "source": "Unavailable",
            "status": f"No valid packet received from ESP32 on {port}",
            "connection_state": "Error",
            "port": port,
            "baudrate": baudrate,
            "data_age": None,
            "timestamp": None,
            "raw_line": last_raw_line,
            "error": "No valid JSON telemetry packet received within timeout."
        }

    except Exception as e:
        return {
            "temperature_c": None,
            "humidity_pct": None,
            "soil_moisture_pct": None,
            "source": "Unavailable",
            "status": f"Disconnected ({port})",
            "connection_state": "Disconnected",
            "port": port,
            "baudrate": baudrate,
            "data_age": None,
            "timestamp": None,
            "raw_line": None,
            "error": f"ESP32 on {port} unavailable: {e}"
        }
    finally:
        if esp32_conn and esp32_conn.is_open:
            try:
                esp32_conn.close()
            except Exception:
                pass


def get_sensor_data(
    mode: str = "simulation",
    port: Optional[str] = None,
    baudrate: int = DEFAULT_BAUD,
    stale_timeout: float = DEFAULT_STALE_TIMEOUT,
    use_sync: bool = False
) -> Dict[str, Any]:
    """
    High-level entry point for fetching sensor readings.

    Supported modes:
      - 'simulation' or 'Simulation Mode': returns synthetic data clearly tagged as 'Simulation'.
      - 'live', 'esp32', or 'Live Sensor Mode': reads from physical ESP32.
        NEVER silently falls back to simulation!

    Returns:
        {
            "temperature_c": float or None,
            "humidity_pct": float or None,
            "soil_moisture_pct": float or None,
            "source": "ESP32" | "Simulation" | "Unavailable",
            "status": str,
            "connection_state": "Connected" | "Disconnected" | "Connecting" | "Error" | "Stale",
            "port": str or None,
            "baudrate": int,
            "data_age": float or None,
            "timestamp": str or None,
            "raw_line": str or None,
            "error": Optional[str]
        }
    """
    clean_mode = mode.strip().lower() if mode else "simulation"

    if clean_mode in ["simulation", "simulation mode", "sim"]:
        return get_simulated_sensor_data()

    # Live Sensor Mode (no simulation fallback permitted under any circumstances)
    target_port = port or DEFAULT_PORT

    if use_sync:
        return read_from_esp32_sync(port=target_port, baudrate=baudrate)

    mgr = SerialManager.get_instance()
    # If port specified differs or not connected, connect
    with mgr.data_lock:
        is_open = mgr.serial_conn is not None and mgr.serial_conn.is_open
        cur_port = mgr.port
        cur_baud = mgr.baudrate

    if not is_open or cur_port != target_port or cur_baud != baudrate:
        mgr.connect(port=target_port, baudrate=baudrate)

    return mgr.get_reading(stale_timeout=stale_timeout)


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
    elif not TEMP_MIN <= temperature <= TEMP_MAX:
        errors.append(f"Temperature reading ({temperature}°C) is outside expected range ({TEMP_MIN}°C to {TEMP_MAX}°C).")

    # Air Humidity validation (0% to 100%)
    if humidity is not None and not HUM_MIN <= humidity <= HUM_MAX:
        errors.append(f"Air humidity ({humidity}%) is outside valid range ({HUM_MIN}% to {HUM_MAX}%).")

    # Soil moisture validation (0% to 100%)
    if soil_moisture is None:
        errors.append("Soil moisture reading unavailable.")
    elif not SOIL_MIN <= soil_moisture <= SOIL_MAX:
        errors.append(f"Soil moisture ({soil_moisture}%) is outside valid range ({SOIL_MIN}% to {SOIL_MAX}%).")

    return errors


def analyze_environment(
    temperature: Optional[float],
    soil_moisture: Optional[float],
    humidity: Optional[float] = None
) -> Tuple[List[str], str]:
    """
    Provides contextual microclimate observations based on sensor measurements.

    Evaluation thresholds:
      - Temperature (°C): <15 (cool), 15-30 (moderate/optimal), >30 (heat stress)
      - Air Humidity (% RH): <40 (dry), 40-75 (optimal canopy development), >75 (high fungal risk)
      - Soil Moisture (%): <30 (dry rootzone), 30-70 (optimal moisture), >70 (saturated/waterlogged)
    """
    observations = []

    if temperature is None and soil_moisture is None and humidity is None:
        return (
            ["Environmental sensor telemetry is currently unavailable."],
            "No physical sensor readings recorded. Prediction is based solely on leaf visual symptoms."
        )

    # Temperature contextual assessment (DHT22)
    if temperature is not None:
        if temperature < 15.0:
            observations.append("Measured temperature is relatively low (< 15°C). Cooler microclimates slow vegetative growth.")
        elif temperature <= 30.0:
            observations.append("Measured temperature is within a moderate range (15°C–30°C), optimal for general crop development.")
        else:
            observations.append("Measured temperature is relatively high (> 30°C). Heat stress can exacerbate wilt and foliar necrosis.")
    else:
        observations.append("Temperature sensor data is unavailable.")

    # Air Humidity contextual assessment (DHT22)
    if humidity is not None:
        if humidity < 40.0:
            observations.append("Air humidity is relatively low (< 40%). Dry atmospheric conditions reduce foliar fungal germination.")
        elif humidity <= 75.0:
            observations.append("Air humidity is optimal for general crop canopy development (40%–75%).")
        else:
            observations.append("High air humidity (> 75%). Prolonged leaf wetness and high RH strongly favor fungal and bacterial spore germination.")
    else:
        observations.append("Air humidity sensor data is unavailable.")

    # Soil moisture contextual assessment (HW-080 on GPIO 34)
    if soil_moisture is not None:
        if soil_moisture < 30.0:
            observations.append("Measured soil moisture is relatively low (< 30%). Dry soil conditions may induce moisture stress.")
        elif soil_moisture <= 70.0:
            observations.append("Measured soil moisture is within a moderate, healthy range (30%–70%).")
        else:
            observations.append("Measured soil moisture is relatively high (> 70%). Soil is near saturation; monitor for root asphyxiation.")
    else:
        observations.append("Soil moisture sensor data is unavailable.")

    overall = (
        "The DHT22 and HW-080 physical sensor readings provide microclimate context for the image-based "
        "disease prediction. They indicate environmental conduciveness and should not be interpreted as "
        "biological proof of pathogen presence."
    )

    return observations, overall