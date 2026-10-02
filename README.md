# 🌱 Plant & Crop Disease Detection and Environmental Monitoring System

An integrated College Capstone Machine Learning, Computer Vision, and IoT system for agricultural plant health diagnostics and environmental condition monitoring.

The system combines **deep learning leaf disease classification** (using a pretrained **EfficientNet-B4** model) with **real-time environmental telemetry** (Temperature and Soil Moisture) received from an **ESP32 microcontroller**, providing environmental context alongside image-based predictions.

---

## 1. Project Overview

Early detection of agricultural crop diseases is essential for food security, minimizing crop yield loss, and reducing unnecessary chemical pesticide usage. This project provides an accessible, end-to-end plant health analysis tool designed for farmers, agronomists, and academic research.

Key capabilities:
- **Image-based Leaf Diagnosis**: Classifies leaf photographs into 38 distinct crop-disease categories.
- **Top-5 Diagnostic Candidates**: Delivers primary predictions and ranked alternative candidates with calibrated confidence scores.
- **Microclimate & Soil Context**: Captures real-time ambient temperature and volumetric soil moisture via physical ESP32 sensors.
- **Environmental Context Analysis**: Evaluates whether current ambient conditions fall into low, moderate, or high physiological ranges.
- **Zero-Cloud Local Persistence**: Stores comprehensive inspection logs in an embedded SQLite database (`crop_disease.db`), eliminating external database setup.
- **Professional Web Interface**: An interactive Streamlit dashboard designed for live demonstrations, viva examinations, and field deployment.

---

## 2. Features

- **Dual Telemetry Workflow**: Combines visual symptom classification with physical environmental measurements.
- **Pretrained Transfer Learning**: Leverages an EfficientNet-B4 architecture fine-tuned on the PlantVillage benchmark.
- **Robust Serial Communication**: Handles ESP32 connection drops, port changes, and data errors without application crashes.
- **Dual Hardware / Simulation Mode**: Supports physical ESP32 hardware streaming or clearly labelled offline simulation for demonstrations.
- **Comprehensive History & Export**: Filter past inspections, inspect individual records, and export detailed printable `.txt` reports.
- **Minimal Dependencies**: Runs purely on standard PyTorch, Streamlit, and Python's built-in SQLite engine.

---

## 3. System Architecture

```text
┌────────────────────────┐         ┌────────────────────────┐
│   Leaf Image Input     │         │   Physical Sensors     │
│  (Upload / Sample)     │         │  (Temp + Soil Moisture)│
└───────────┬────────────┘         └───────────┬────────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────┐         ┌────────────────────────┐
│  Image Preprocessing   │         │   ESP32 Microcontroller│
│  (Resize 256, Crop 224)│         │   (Newline JSON Serial)│
└───────────┬────────────┘         └───────────┬────────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────┐         ┌────────────────────────┐
│ EfficientNet-B4 Model  │         │   Sensor Subsystem     │
│ (38 PlantVillage Cls)  │         │   (Validation & Range) │
└───────────┬────────────┘         └───────────┬────────────┘
            │                                  │
            └─────────────────┬────────────────┘
                              ▼
                ┌───────────────────────────┐
                │  Streamlit Web Dashboard  │
                │  - Prediction & Confidence│
                │  - Top-5 Candidates       │
                │  - Environmental Context  │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │  Local SQLite Persistence │
                │     (crop_disease.db)     │
                └───────────────────────────┘
```

---

## 4. Machine Learning Model

- **Architecture**: `EfficientNet-B4` with custom 2-stage classification head.
- **Pretrained Checkpoint**: `Khawajaa/plant-disease-detector` (`best_model.pth`).
- **Base Framework**: PyTorch (`torch`, `torchvision`).
- **Input Resolution**: $3 \times 224 \times 224$ (RGB).
- **Classification Head**:
  ```python
  nn.Sequential(
      nn.Dropout(p=0.4),
      nn.Linear(1792, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.2),
      nn.Linear(512, 38)
  )
  ```
- **Validation Accuracy**: 98.87% on benchmark test split.
- **Input Preprocessing**:
  1. `Resize(256)`
  2. `CenterCrop(224)`
  3. `ToTensor()`
  4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 5. Dataset & Classes

The model classifies **38 distinct classes** from the **PlantVillage** dataset spanning 14 crop species:

- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn (Maize)**: Cercospora leaf spot (Gray leaf spot), Common rust, Northern leaf blight, Healthy
- **Grape**: Black rot, Esca (Black measles), Leaf blight (Isariopsis), Healthy
- **Orange**: Citrus greening (Huanglongbing)
- **Peach**: Bacterial spot, Healthy
- **Pepper, Bell**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy

---

## 6. Hardware Specifications

- **Microcontroller**: ESP32 Dev Module (NodeMCU ESP32 / ESP-WROOM-32).
- **Sensors**:
  1. **Temperature Sensor**: Analog (LM35 / TMP36) or Digital (DS18B20).
  2. **Soil Moisture Sensor**: Capacitive Soil Moisture Sensor v1.2 (or Resistive Soil Hygrometer).
- **Important Distinction**: Soil moisture measures volumetric soil water content, **not** ambient air humidity. No humidity sensor is assumed.

---

## 7. Software Requirements

- Python 3.10 – 3.12 (Recommended: Python 3.12)
- Virtual environment (`venv` or `conda`)
- Modern web browser (Chrome, Edge, Firefox)

---

## 8. Installation

1. **Clone or Navigate to the Repository**:
   ```bash
   cd "Crop Disease Detection"
   ```

2. **Create and Activate a Virtual Environment**:
   ```powershell
   # Windows PowerShell (If script execution is disabled, allow it for the current process):
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

   *(Note: The pretrained `best_model.pth` (74.7 MB) is automatically verified on startup and downloaded from Hugging Face if not already present in the `model/` folder).*

---

## 9. Running the Application

### Option A: Using the Windows Launcher (Easiest)
Simply double-click:
- **`run_app.bat`**: Launches the Streamlit Web Application directly.
- **`run_tests.bat`**: Runs the 10-step automated validation suite.

### Option B: Direct Execution (No Activation Needed)
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### Option C: After Virtualenv Activation
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The application will start and open automatically in your browser at `http://localhost:8501`.


To run on a specific port or network address:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 10. ESP32 Setup & Wiring

### Hardware Connections
| ESP32 Pin | Sensor Component | Note |
| :--- | :--- | :--- |
| `GPIO 34` (ADC6) | Soil Moisture `AOUT` | Analog output from soil probe |
| `GPIO 35` (ADC7) | Temperature Sensor Output | Analog signal from temp sensor |
| `3V3` or `5V` | Sensor `VCC` | Power rail |
| `GND` | Sensor `GND` | Common ground |

### Arduino Firmware Sketch
Flash the following sketch to your ESP32 using the Arduino IDE:

```cpp
#include <Arduino.h>

const int SOIL_PIN = 34; // Capacitive Soil Moisture Sensor Analog Pin
const int TEMP_PIN = 35; // Analog Temperature Sensor Pin

void setup() {
  Serial.begin(115200);
  delay(1000);
}

void loop() {
  int raw_soil = analogRead(SOIL_PIN);
  int raw_temp = analogRead(TEMP_PIN);

  // Calibrate mapping to your specific sensor bounds
  // Example for Capacitive Sensor: 3000 (air/dry) -> 1200 (water/wet)
  float soil_moisture = map(raw_soil, 3200, 1200, 0, 100);
  soil_moisture = constrain(soil_moisture, 0.0, 100.0);

  // Example LM35 temperature conversion (10mV per degree C)
  float voltage = (raw_temp / 4095.0) * 3.3;
  float temperature = voltage * 100.0;

  // Emit newline-delimited JSON
  Serial.print("{\"temperature\":");
  Serial.print(temperature, 1);
  Serial.print(",\"soil_moisture\":");
  Serial.print(soil_moisture, 1);
  Serial.println("}");

  delay(2000); // Sample every 2 seconds
}
```

---

## 11. Sensor Data Format

The Python application expects newline-delimited JSON over serial at **115200 baud**:

```json
{"temperature": 28.4, "soil_moisture": 68.0}
```

The Python application parses and validates this into:
```python
{
    "temperature_c": 28.4,
    "soil_moisture_pct": 68.0,
    "source": "ESP32",
    "status": "Connected to ESP32 on COM3"
}
```

### Environmental Context Ranges:
- **Temperature**:
  - $< 15^\circ\text{C}$: Relatively low
  - $15^\circ\text{C} - 30^\circ\text{C}$: Moderate range
  - $> 30^\circ\text{C}$: Relatively high
- **Soil Moisture**:
  - $< 30\%$: Relatively low
  - $30\% - 70\%$: Moderate range
  - $> 70\%$: Relatively high

*Context Disclaimer: Physical sensor readings provide ambient context for leaf evaluation and do not represent biological proof of pathogen presence.*

---

## 12. SQLite Database

All MongoDB dependencies have been fully removed and replaced with Python's built-in `sqlite3` engine.

- **Database File**: `crop_disease.db` (auto-created on startup)
- **Table Name**: `analyses`
- **Schema**:
  | Column | Type | Description |
  | :--- | :--- | :--- |
  | `id` | INTEGER PRIMARY KEY | Unique auto-incrementing record ID |
  | `timestamp` | TEXT | ISO / formatted date-time string |
  | `image_name` | TEXT | Source image filename |
  | `plant` | TEXT | Common name of target crop (e.g. Tomato) |
  | `predicted_condition` | TEXT | Predicted disease condition or Healthy |
  | `predicted_class` | TEXT | Full PlantVillage class identifier |
  | `confidence` | REAL | Model probability score (0.0 to 1.0) |
  | `temperature_c` | REAL | Temperature at time of analysis |
  | `soil_moisture_pct` | REAL | Soil moisture at time of analysis |
  | `sensor_source` | TEXT | Origin: `ESP32` or `Simulation` |
  | `top_predictions_json`| TEXT | JSON list of top-5 predictions |
  | `environmental_observations_json` | TEXT | JSON list of contextual notes |
  | `environmental_summary` | TEXT | Contextual explanation string |

---

## 13. Project Structure

```text
Crop Disease Detection/
│
├── app.py                     # Streamlit multi-tab web application
├── model.py                   # EfficientNet-B4 classifier & HF downloader
├── sensors.py                 # ESP32 serial communication & validation
├── database.py                # Local SQLite persistence (crop_disease.db)
├── utils.py                   # Report generator, synthetic leaf, helpers
├── test_pipeline.py           # Comprehensive 10-step test suite
│
├── model/
│   └── best_model.pth         # EfficientNet-B4 weights checkpoint (74.7 MB)
│
├── notebooks/
│   └── CropDiseaseMDM.ipynb   # Original research & experimentation notebook
│
├── crop_disease.db            # Embedded SQLite database
├── requirements.txt           # Minimal, clean project dependencies
├── .gitignore                 # Git ignore rules for venv, cache, etc.
└── README.md                  # Complete project documentation
```

---

## 14. Limitations

1. **Dataset Bias**: The model was trained on the PlantVillage laboratory dataset, which features isolated leaves against uniform backgrounds. Complex field scenes with background clutter or multiple overlapping leaves may require image cropping.
2. **Environmental Independence**: Sensor readings provide ambient microclimate context; the vision model classifier does not ingest sensor telemetry directly into its feature vector.
3. **No Direct Pathogen Proof**: Environmental context indicates whether temperature or soil moisture favors fungal or bacterial proliferation, but does not substitute for laboratory microbial assays.

---

## 15. Future Improvements

- **Multimodal Fusion**: Train a multimodal deep network that concatenates numerical sensor embeddings directly with EfficientNet visual features before the classification head.
- **Edge Deployment**: Quantize the model using PyTorch Mobile or ONNX Runtime for real-time inference on edge devices (Raspberry Pi 5 / Jetson Nano).
- **Grad-CAM Saliency Maps**: Integrate Class Activation Maps to visually highlight the diseased regions on the leaf.
- **Automated Treatment Advisory**: Connect localized treatment suggestions (organic or chemical fungicides) mapped to detected diseases.

---

## 16. Verification & Testing

To execute the automated 10-stage test suite covering model inference, serial fallback, SQLite persistence, and dependency audits:

```bash
python test_pipeline.py
```
