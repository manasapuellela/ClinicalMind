"""
DELTA LAKE WRITER (Spark pipeline only — not used on Streamlit Cloud).
Saves the fully processed and quality-checked patient DataFrame
to Delta Lake format in data/processed/. Also saves a JSON summary.
The app and agent load JSON via pipeline.patient_loader (Spark-free).
"""

import os
import json
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


DELTA_PATH = "data/processed/patients_delta"
JSON_PATH = "data/processed/patients_summary.json"


def write_to_delta(df: DataFrame):
    """
    Writes the processed DataFrame to Delta Lake.
    Overwrites on each pipeline run.
    """
    os.makedirs("data/processed", exist_ok=True)
    
    print(f"Writing {df.count()} records to Delta Lake at {DELTA_PATH}...")
    
    df.write.format("delta").mode("overwrite").save(DELTA_PATH)
    print("Delta Lake write complete.")


def write_json_summary(df: DataFrame):
    """
    Writes a lightweight JSON summary of all patient records
    for the RAG agent to load without needing a Spark session.
    Includes only scoreable records.
    """
    scoreable_df = df.filter(col("is_scoreable") == True)
    
    # Convert to pandas for JSON export
    pandas_df = scoreable_df.select(
        "patient_id", "age", "gender", "diagnosis",
        "length_of_stay", "prior_admissions", "has_follow_up",
        "lives_alone", "non_compliant", "medications_raw",
        "completeness_score", "confidence_label", "quality_warning"
    ).toPandas()
    
    # Convert to list of dicts
    records = pandas_df.to_dict(orient="records")
    
    with open(JSON_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)
    
    print(f"JSON summary written: {len(records)} scoreable records saved to {JSON_PATH}")
    return records


def load_patients_json():
    """
    Loads the JSON summary for use by the agent layer.
    No Spark needed — plain Python.
    """
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"No processed data found at {JSON_PATH}. "
            "Run 'python run_pipeline.py' first."
        )
    with open(JSON_PATH, "r") as f:
        return json.load(f)

