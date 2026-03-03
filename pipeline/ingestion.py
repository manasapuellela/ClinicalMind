"""
INGESTION NODE
Reads all raw .txt discharge summaries from data/raw/
Uses PySpark to load them into a DataFrame.
Each row = one patient document.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, col, lit
from pyspark.sql.types import StructType, StructField, StringType


def get_spark():
    """Initialize a local SparkSession for pipeline processing."""
    return (
        SparkSession.builder
        .appName("ClinicalMind-Pipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
        .master("local[*]")
        .getOrCreate()
    )


def ingest_raw_documents(spark, raw_dir="data/raw"):
    """
    Reads all .txt files from raw_dir into a Spark DataFrame.
    Returns a DataFrame with columns: file_name, raw_text
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    txt_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    if not txt_files:
        raise ValueError(f"No .txt files found in {raw_dir}. Run generate_data.py first.")
    
    print(f"Found {len(txt_files)} discharge summaries. Loading into Spark...")
    
    # Read all text files — each file becomes one row
    df = spark.read.text(raw_dir, wholetext=True)
    df = df.withColumnRenamed("value", "raw_text")
    df = df.withColumn("file_name", input_file_name())
    df = df.withColumn(
        "file_name",
        regexp_extract(col("file_name"), r"([^/\\]+)$", 1)
    )
    
    print(f"Ingested {df.count()} documents into Spark DataFrame.")
    return df

