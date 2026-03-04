"""
PATIENT LOADER — Spark-free.
Loads patients_summary.json for the agent/app. Use this in the app and agent
so Cloud deployment never imports PySpark (delta_writer does).
"""

import os
import json

JSON_PATH = "data/processed/patients_summary.json"


def load_patients_json():
    """
    Loads the JSON summary for use by the agent layer.
    No Spark needed — plain Python.
    """
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"No processed data found at {JSON_PATH}. "
            "Run 'python run_pipeline.py' first or let the app bootstrap."
        )
    with open(JSON_PATH, "r") as f:
        return json.load(f)
