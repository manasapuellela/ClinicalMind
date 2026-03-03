"""
FALLBACK PIPELINE — Pure Python (no Spark).
Use when Spark fails on Windows (Hadoop native IO / Delta classpath).
Reads raw .txt, extracts fields, computes quality, writes JSON only.
Same output schema as the Spark path for load_patients_json().
"""

import os
import re
import json

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
JSON_PATH = "data/processed/patients_summary.json"

# Same weights as quality_check.py
CRITICAL_FIELDS = {
    "patient_id":       {"weight": 10, "label": "Patient ID"},
    "age":              {"weight": 20, "label": "Age"},
    "diagnosis":        {"weight": 25, "label": "Primary Diagnosis"},
    "length_of_stay":   {"weight": 15, "label": "Length of Stay"},
    "prior_admissions": {"weight": 20, "label": "Prior Admissions"},
    "medications_raw":  {"weight": 10, "label": "Medications"},
}


def _extract_one(raw_text: str) -> dict:
    """Extract structured fields from one discharge summary (mirrors extractor.py)."""
    text_lower = raw_text.lower()

    m = re.search(r"Patient ID:\s*(PAT-\d+)", raw_text, re.I)
    patient_id = m.group(1) if m else None

    m = re.search(r"Age:\s*(\d+|Unknown)", raw_text, re.I)
    age = int(m.group(1)) if m and m.group(1).isdigit() else None

    m = re.search(r"Gender:\s*(Male|Female)", raw_text, re.I)
    gender = m.group(1) if m else None

    m = re.search(r"Primary Diagnosis:\s*(.+)", raw_text)
    diagnosis = None
    if m:
        d = m.group(1).strip()
        if "not documented" not in d.lower():
            diagnosis = d

    m = re.search(r"Length of Stay:\s*(\d+)", raw_text)
    length_of_stay = int(m.group(1)) if m else None

    m = re.search(r"Prior Admissions.*?:\s*(\d+)", raw_text)
    prior_admissions = int(m.group(1)) if m else None

    has_follow_up = "no follow-up appointment scheduled" not in text_lower and "follow-up" in text_lower
    lives_alone = "lives alone" in text_lower
    non_compliant = "non-compliance" in text_lower

    m = re.search(r"Medications at Discharge:\s*(.+)", raw_text)
    medications_raw = None
    if m:
        med = m.group(1).strip()
        if "see pharmacy" not in med.lower():
            medications_raw = med

    return {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "diagnosis": diagnosis,
        "length_of_stay": length_of_stay,
        "prior_admissions": prior_admissions,
        "has_follow_up": has_follow_up,
        "lives_alone": lives_alone,
        "non_compliant": non_compliant,
        "medications_raw": medications_raw,
    }


def _quality_for_record(record: dict) -> dict:
    """Add completeness_score, confidence_label, is_scoreable, quality_warning (mirrors quality_check)."""
    score = 0.0
    missing = []
    for field, meta in CRITICAL_FIELDS.items():
        val = record.get(field)
        if val is not None and str(val).strip() != "":
            score += meta["weight"]
        else:
            missing.append(meta["label"])

    record["completeness_score"] = round(score, 1)
    record["confidence_label"] = "HIGH" if score >= 80 else ("MEDIUM" if score >= 50 else "LOW")
    record["is_scoreable"] = score >= 50
    record["quality_warning"] = (
        "⚠️ Record incomplete. Risk score unreliable. Manual review required."
        if record["confidence_label"] == "LOW"
        else (
            "⚠️ Some fields missing. Risk score is approximate."
            if record["confidence_label"] == "MEDIUM"
            else "✅ Record complete. Risk score reliable."
        )
    )
    return record


def run_fallback_pipeline():
    """Run ingestion + extraction + quality + JSON write without Spark."""
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    txt_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
    if not txt_files:
        raise ValueError(f"No .txt files in {RAW_DIR}. Run generate_data.py first.")

    print(f"Fallback pipeline: reading {len(txt_files)} files (no Spark)...")
    records = []
    for fname in txt_files:
        path = os.path.join(RAW_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        rec = _extract_one(raw_text)
        rec = _quality_for_record(rec)
        records.append(rec)

    scoreable = [r for r in records if r.get("is_scoreable")]
    # Drop is_scoreable for JSON output; keep same columns as Spark path
    out_cols = [
        "patient_id", "age", "gender", "diagnosis", "length_of_stay",
        "prior_admissions", "has_follow_up", "lives_alone", "non_compliant",
        "medications_raw", "completeness_score", "confidence_label", "quality_warning"
    ]
    summary = [{k: r.get(k) for k in out_cols} for r in scoreable]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    high = sum(1 for r in scoreable if r["confidence_label"] == "HIGH")
    med = sum(1 for r in scoreable if r["confidence_label"] == "MEDIUM")
    low = sum(1 for r in records if not r.get("is_scoreable"))

    print(f"Quality: HIGH={high}, MEDIUM={med}, LOW={low}")
    print(f"JSON summary written: {len(summary)} scoreable records to {JSON_PATH}")
    return summary
