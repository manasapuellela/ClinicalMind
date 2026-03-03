"""
DATA QUALITY LAYER — The differentiator of this project.

Every patient record gets:
1. A completeness score (0-100)
2. A list of missing or problematic fields
3. A confidence label: HIGH / MEDIUM / LOW
4. A scoreable flag: can we trust the AI risk score for this record?

This is what makes the AI trustworthy — it knows what it doesn't know.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, udf, array, array_remove, concat_ws, round as spark_round
)
from pyspark.sql.types import StringType, FloatType, ArrayType, BooleanType


# Fields required for a reliable risk score — and their weights
CRITICAL_FIELDS = {
    "patient_id":       {"weight": 10, "label": "Patient ID"},
    "age":              {"weight": 20, "label": "Age"},
    "diagnosis":        {"weight": 25, "label": "Primary Diagnosis"},
    "length_of_stay":   {"weight": 15, "label": "Length of Stay"},
    "prior_admissions": {"weight": 20, "label": "Prior Admissions"},
    "medications_raw":  {"weight": 10, "label": "Medications"},
}


def compute_quality_scores(df: DataFrame) -> DataFrame:
    """
    Adds data quality columns to each patient record.
    """
    
    # --- Completeness Score ---
    # Each present field contributes its weight to the total score
    score_expr = lit(0.0)
    missing_flags = []
    
    for field, meta in CRITICAL_FIELDS.items():
        is_present = col(field).isNotNull() & (col(field).cast(StringType()) != "")
        score_expr = score_expr + when(is_present, lit(float(meta["weight"]))).otherwise(lit(0.0))
        missing_flags.append(
            when(~is_present, lit(meta["label"])).otherwise(lit(None).cast(StringType()))
        )
    
    df = df.withColumn("completeness_score", spark_round(score_expr, 1))
    
    # --- Missing Fields List ---
    df = df.withColumn(
        "missing_fields",
        concat_ws(", ", *[
            when(col(field).isNull() | (col(field).cast(StringType()) == ""), lit(meta["label"]))
            .otherwise(lit(""))
            for field, meta in CRITICAL_FIELDS.items()
        ])
    )
    
    # Clean up empty strings from missing_fields
    df = df.withColumn(
        "missing_fields",
        when(col("missing_fields") == "", lit("None")).otherwise(col("missing_fields"))
    )
    
    # --- Confidence Label ---
    df = df.withColumn(
        "confidence_label",
        when(col("completeness_score") >= 80, lit("HIGH"))
        .when(col("completeness_score") >= 50, lit("MEDIUM"))
        .otherwise(lit("LOW"))
    )
    
    # --- Scoreable Flag ---
    # Only HIGH and MEDIUM confidence records get AI risk scoring
    df = df.withColumn(
        "is_scoreable",
        when(col("completeness_score") >= 50, lit(True)).otherwise(lit(False))
    )
    
    # --- Quality Warning Message ---
    df = df.withColumn(
        "quality_warning",
        when(
            col("confidence_label") == "LOW",
            lit("⚠️ Record incomplete. Risk score unreliable. Manual review required.")
        ).when(
            col("confidence_label") == "MEDIUM",
            lit("⚠️ Some fields missing. Risk score is approximate.")
        ).otherwise(lit("✅ Record complete. Risk score reliable."))
    )
    
    print(f"Quality check complete.")
    print(f"  HIGH confidence:   {df.filter(col('confidence_label') == 'HIGH').count()}")
    print(f"  MEDIUM confidence: {df.filter(col('confidence_label') == 'MEDIUM').count()}")
    print(f"  LOW confidence:    {df.filter(col('confidence_label') == 'LOW').count()}")
    
    return df

