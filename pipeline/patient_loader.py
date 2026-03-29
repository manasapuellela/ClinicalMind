"""
PATIENT LOADER — Spark-free.
Loads patients_summary.json for the agent/app. Use this in the app and agent
so Cloud deployment never imports PySpark (delta_writer does).
"""

import os
import json

from pipeline.schemas import ValidatedPatientBatch

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


def load_patients_validated() -> ValidatedPatientBatch:
    """
    Loads patients_summary.json and validates each row with Pydantic.
    """
    raw = load_patients_json()
    batch = ValidatedPatientBatch.from_raw_list(raw)
    n_passed = batch.total
    n_errors = len(batch.validation_errors)
    print(f"Validation complete: {n_passed} records passed, {n_errors} errors")
    return batch


def validate_and_load() -> list[dict]:
    """
    Same as load_patients_json() but runs Pydantic validation; returns plain dicts.
    Drop-in when callers want validated records only (invalid rows omitted).
    """
    batch = load_patients_validated()
    return [r.model_dump() for r in batch.records]
