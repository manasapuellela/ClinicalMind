"""
EXTRACTION NODE
Takes the raw text DataFrame from ingestion.
Extracts structured fields from unstructured text using regex patterns.
Returns a DataFrame with clean, structured patient columns.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    regexp_extract, col, trim, when, lower, udf
)
from pyspark.sql.types import IntegerType, StringType
import re


def extract_fields(df: DataFrame) -> DataFrame:
    """
    Extracts structured fields from raw discharge summary text.
    Uses regex patterns to pull out key clinical fields.
    """
    
    # Patient ID
    df = df.withColumn(
        "patient_id",
        regexp_extract(col("raw_text"), r"Patient ID:\s*(PAT-\d+)", 1)
    )
    
    # Age — extract numeric value
    df = df.withColumn(
        "age_raw",
        regexp_extract(col("raw_text"), r"Age:\s*(\d+|Unknown)", 1)
    )
    df = df.withColumn(
        "age",
        when(col("age_raw").rlike(r"^\d+$"), col("age_raw").cast(IntegerType()))
        .otherwise(None)
    )
    
    # Gender
    df = df.withColumn(
        "gender",
        regexp_extract(col("raw_text"), r"Gender:\s*(Male|Female)", 1)
    )
    
    # Primary diagnosis
    df = df.withColumn(
        "diagnosis",
        regexp_extract(col("raw_text"), r"Primary Diagnosis:\s*(.+)", 1)
    )
    df = df.withColumn(
        "diagnosis",
        when(
            lower(col("diagnosis")).contains("not documented"), None
        ).otherwise(trim(col("diagnosis")))
    )
    
    # Length of stay
    df = df.withColumn(
        "length_of_stay",
        regexp_extract(col("raw_text"), r"Length of Stay:\s*(\d+)", 1).cast(IntegerType())
    )
    
    # Prior admissions
    df = df.withColumn(
        "prior_admissions",
        regexp_extract(col("raw_text"), r"Prior Admissions.*?:\s*(\d+)", 1).cast(IntegerType())
    )
    
    # Has follow-up appointment
    df = df.withColumn(
        "has_follow_up",
        when(
            lower(col("raw_text")).contains("no follow-up appointment scheduled"), False
        ).when(
            lower(col("raw_text")).contains("follow-up"), True
        ).otherwise(False)
    )
    
    # Lives alone flag
    df = df.withColumn(
        "lives_alone",
        when(
            lower(col("raw_text")).contains("lives alone"), True
        ).otherwise(False)
    )
    
    # Medication non-compliance flag
    df = df.withColumn(
        "non_compliant",
        when(
            lower(col("raw_text")).contains("non-compliance"), True
        ).otherwise(False)
    )
    
    # Medications — extract as raw string
    df = df.withColumn(
        "medications_raw",
        regexp_extract(col("raw_text"), r"Medications at Discharge:\s*(.+)", 1)
    )
    df = df.withColumn(
        "medications_raw",
        when(
            lower(col("medications_raw")).contains("see pharmacy"), None
        ).otherwise(trim(col("medications_raw")))
    )
    
    # Drop raw columns
    df = df.drop("age_raw")
    
    print(f"Extraction complete. Structured {df.count()} patient records.")
    return df

