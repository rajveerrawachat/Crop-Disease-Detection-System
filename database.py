"""
SQLite Database Module for Crop Disease Detection & Environmental Monitoring
Replaces legacy MongoDB implementation with Python standard library sqlite3.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent / "crop_disease.db"


def get_connection() -> sqlite3.Connection:
    """
    Returns a connection to the SQLite database with Row row_factory.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """
    Initializes the SQLite database and creates/migrates the analyses table.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT,
                plant TEXT NOT NULL,
                predicted_condition TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                soil_moisture_pct REAL,
                sensor_source TEXT DEFAULT 'Unavailable',
                top_predictions_json TEXT,
                environmental_observations_json TEXT,
                environmental_summary TEXT
            )
        """)
        
        # Check if humidity_pct column exists in older database tables and add if missing
        cursor.execute("PRAGMA table_info(analyses)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "humidity_pct" not in columns:
            cursor.execute("ALTER TABLE analyses ADD COLUMN humidity_pct REAL")

        conn.commit()


def save_analysis(data: Dict[str, Any]) -> int:
    """
    Saves an analysis record into the analyses table.
    Accepts either a flat dictionary or a nested dictionary containing 'prediction' and 'environment'.
    Returns the newly inserted record id.
    """
    init_database()

    # Extract fields supporting both nested and flat dictionaries
    timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_name = data.get("image_name") or data.get("image") or "uploaded_leaf.jpg"

    prediction = data.get("prediction", {})
    environment = data.get("environment", {})

    plant = data.get("plant") or prediction.get("plant") or "Unknown"
    condition = data.get("predicted_condition") or data.get("condition") or prediction.get("condition") or "Unknown"
    predicted_class = data.get("predicted_class") or prediction.get("predicted_class") or f"{plant}___{condition}"
    confidence = float(data.get("confidence") or prediction.get("confidence") or 0.0)

    temperature_c = (
        data.get("temperature_c")
        if data.get("temperature_c") is not None
        else environment.get("temperature_c")
    )
    humidity_pct = (
        data.get("humidity_pct")
        if data.get("humidity_pct") is not None
        else environment.get("humidity_pct")
    )
    soil_moisture_pct = (
        data.get("soil_moisture_pct")
        if data.get("soil_moisture_pct") is not None
        else environment.get("soil_moisture_pct")
    )
    sensor_source = (
        data.get("sensor_source")
        or environment.get("source")
        or "Unavailable"
    )

    # Serialize top predictions if present
    top_preds = data.get("top_predictions") or prediction.get("top_predictions")
    top_predictions_json = json.dumps(top_preds) if top_preds is not None else None

    # Serialize observations if present
    obs = data.get("environmental_observations") or data.get("observations")
    environmental_observations_json = json.dumps(obs) if obs is not None else None

    environmental_summary = data.get("environmental_summary") or data.get("overall")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (
                timestamp,
                image_name,
                plant,
                predicted_condition,
                predicted_class,
                confidence,
                temperature_c,
                humidity_pct,
                soil_moisture_pct,
                sensor_source,
                top_predictions_json,
                environmental_observations_json,
                environmental_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            image_name,
            plant,
            condition,
            predicted_class,
            confidence,
            temperature_c,
            humidity_pct,
            soil_moisture_pct,
            sensor_source,
            top_predictions_json,
            environmental_observations_json,
            environmental_summary
        ))
        conn.commit()
        return cursor.lastrowid


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Converts a sqlite3.Row object to a Python dictionary, parsing JSON fields.
    """
    record = dict(row)
    if record.get("top_predictions_json"):
        try:
            record["top_predictions"] = json.loads(record["top_predictions_json"])
        except (ValueError, TypeError):
            record["top_predictions"] = []
    else:
        record["top_predictions"] = []

    if record.get("environmental_observations_json"):
        try:
            record["environmental_observations"] = json.loads(record["environmental_observations_json"])
        except (ValueError, TypeError):
            record["environmental_observations"] = []
    else:
        record["environmental_observations"] = []

    return record


def get_analysis_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent analyses up to the specified limit.
    """
    init_database()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM analyses
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [_row_to_dict(row) for row in rows]


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single analysis by its primary key ID.
    """
    init_database()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None


def delete_analysis(analysis_id: int) -> bool:
    """
    Deletes an analysis record by ID.
    """
    init_database()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_history() -> int:
    """
    Deletes all records from the analyses table.
    Returns the number of deleted rows.
    """
    init_database()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses")
        conn.commit()
        return cursor.rowcount


def get_statistics() -> Dict[str, Any]:
    """
    Computes aggregate metrics for dashboard analytics.
    """
    init_database()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analyses")
        total_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM analyses
            WHERE LOWER(predicted_condition) LIKE '%healthy%'
        """)
        healthy_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT plant, COUNT(*) as cnt FROM analyses
            GROUP BY plant ORDER BY cnt DESC LIMIT 5
        """)
        top_plants = [{"plant": row[0], "count": row[1]} for row in cursor.fetchall()]

        cursor.execute("""
            SELECT predicted_condition, COUNT(*) as cnt FROM analyses
            WHERE LOWER(predicted_condition) NOT LIKE '%healthy%'
            GROUP BY predicted_condition ORDER BY cnt DESC LIMIT 5
        """)
        top_diseases = [{"condition": row[0], "count": row[1]} for row in cursor.fetchall()]

        return {
            "total_analyses": total_count,
            "healthy_count": healthy_count,
            "diseased_count": total_count - healthy_count,
            "top_plants": top_plants,
            "top_diseases": top_diseases
        }