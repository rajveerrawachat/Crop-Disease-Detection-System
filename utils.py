"""
Utility Helper Functions for Crop Disease Detection & Environmental Monitoring
Includes report generation, image validation, confidence interpretation, and sample images.
"""

from typing import Dict, Any, List
from PIL import Image, ImageDraw


def get_confidence_interpretation(confidence: float) -> str:
    """
    Returns an academic and cautious interpretation of the model's confidence probability.
    """
    if confidence < 0.50:
        return "Low model confidence (<50%). Visual symptoms are ambiguous; manual extension inspection recommended."
    elif confidence < 0.75:
        return "Moderate model confidence (50%–75%). Visual symptoms align with candidate class; verify against field context."
    else:
        return "High model confidence (>75%). Leaf visual features strongly correlate with trained class representations."


def generate_sample_leaf_image(plant: str = "Tomato", healthy: bool = True) -> Image.Image:
    """
    Generates a synthetic sample leaf image for quick UI demonstration when no physical photo is handy.
    """
    img = Image.new("RGB", (300, 300), color=(240, 245, 240))
    draw = ImageDraw.Draw(img)

    # Draw a stylized leaf shape
    leaf_color = (46, 125, 50) if healthy else (139, 115, 85)
    vein_color = (27, 94, 32) if healthy else (90, 70, 50)

    # Draw oval leaf body
    draw.ellipse([50, 40, 250, 260], fill=leaf_color, outline=vein_color, width=3)

    # Draw central vein
    draw.line([150, 40, 150, 260], fill=vein_color, width=4)

    # Draw side veins
    for y in range(80, 240, 35):
        draw.line([150, y, 90, y - 25], fill=vein_color, width=2)
        draw.line([150, y, 210, y - 25], fill=vein_color, width=2)

    # If diseased, draw necrosis/blight spots
    if not healthy:
        spot_color = (70, 40, 20)
        spots = [(100, 100, 15), (180, 120, 20), (130, 190, 18), (170, 210, 12)]
        for x, y, r in spots:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=spot_color)

    return img


def generate_text_report(analysis: Dict[str, Any]) -> str:
    """
    Formats the analysis record into a comprehensive printable/exportable text report.
    Clearly distinguishes physical ESP32 telemetry from Simulation data.
    """
    plant = analysis.get("plant", "Unknown")
    condition = analysis.get("predicted_condition") or analysis.get("condition", "Unknown")
    conf = analysis.get("confidence", 0.0)
    temp = analysis.get("temperature_c")
    hum = analysis.get("humidity_pct")
    moist = analysis.get("soil_moisture_pct")
    source = analysis.get("sensor_source") or analysis.get("source", "Unavailable")
    timestamp = analysis.get("timestamp", "N/A")
    image_name = analysis.get("image_name", "N/A")
    top_preds = analysis.get("top_predictions", [])
    obs = analysis.get("environmental_observations", [])
    summary = analysis.get("environmental_summary", "")

    temp_str = f"{temp:.1f} °C" if temp is not None else "Unavailable"
    hum_str = f"{hum:.1f} % RH" if hum is not None else "Unavailable"
    moist_str = f"{moist:.1f} %" if moist is not None else "Unavailable"
    conf_msg = get_confidence_interpretation(conf)

    # Distinct source labeling
    if source == "ESP32":
        source_label = "ESP32 Physical Microcontroller (Live Telemetry: GPIO 4 DHT22 + GPIO 34 HW-080)"
    elif source == "Simulation":
        source_label = "SIMULATED (Offline Demonstration Mode - No Physical Hardware)"
    else:
        source_label = f"Unavailable / Disconnected ({source})"

    top_preds_text = ""
    for i, p in enumerate(top_preds, start=1):
        p_label = p.get("readable") or f"{p.get('plant', '')} — {p.get('condition', '')}"
        p_conf = p.get("confidence", 0.0) * 100
        top_preds_text += f"{i}. {p_label} ({p_conf:.2f}%)\n"

    obs_text = ""
    for o in obs:
        obs_text += f"• {o}\n"

    report = f"""============================================================
           PLANT HEALTH & ENVIRONMENTAL MONITORING REPORT
============================================================

Report Timestamp : {timestamp}
Analyzed Image   : {image_name}

------------------------------------------------------------
IMAGE-BASED DISEASE PREDICTION
------------------------------------------------------------
Target Plant         : {plant}
Predicted Condition  : {condition}
Model Confidence     : {conf * 100:.2f}%
Model Architecture   : EfficientNet-B4 (Khawajaa / PlantVillage 38 Classes)

Confidence Note:
{conf_msg}

------------------------------------------------------------
TOP CANDIDATE PREDICTIONS
------------------------------------------------------------
{top_preds_text if top_preds_text else "No additional candidates recorded.\n"}
------------------------------------------------------------
PHYSICAL SENSOR MEASUREMENTS & TELEMETRY
------------------------------------------------------------
Sensor Source           : {source_label}
Temperature (DHT22)     : {temp_str}
Air Humidity (DHT22)    : {hum_str}
Soil Moisture (HW-080)  : {moist_str}

------------------------------------------------------------
MICROCLIMATE ENVIRONMENTAL CONTEXT
------------------------------------------------------------
{obs_text if obs_text else "No environmental observations recorded.\n"}
Summary Context:
{summary}

"""
    return report
