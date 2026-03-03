"""
ONE COMMAND PIPELINE RUNNER
Runs the full data pipeline in sequence:
  1. Ingest raw discharge summaries
  2. Extract structured fields
  3. Run data quality checks
  4. Write to Delta Lake + JSON summary
  5. Build RAG vector store

Run this before starting the Streamlit app.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def run():
    print("=" * 50)
    print("  ClinicalMind Data Pipeline")
    print("=" * 50)

    # Step 1 — Check raw data exists
    if not os.listdir("data/raw"):
        print("No raw data found. Generating synthetic data...")
        import generate_data
    else:
        print(f"Found {len(os.listdir('data/raw'))} raw files.")

    # Step 2 — Spark pipeline (fallback to Python-only on Windows/Spark failure)
    use_fallback = False
    try:
        print("\n[1/4] Starting Spark session and ingesting documents...")
        from pipeline.ingestion import get_spark, ingest_raw_documents
        spark = get_spark()
        df = ingest_raw_documents(spark, "data/raw")

        print("\n[2/4] Extracting structured fields...")
        from pipeline.extractor import extract_fields
        df = extract_fields(df)

        print("\n[3/4] Running data quality checks...")
        from pipeline.quality_check import compute_quality_scores
        df = compute_quality_scores(df)

        print("\n[4/4] Writing to Delta Lake and JSON summary...")
        from pipeline.delta_writer import write_to_delta, write_json_summary
        write_to_delta(df)
        write_json_summary(df)
    except Exception as e:
        print(f"\nSpark pipeline failed ({e}). Using Python-only fallback (no Delta Lake)...")
        from pipeline.fallback_pipeline import run_fallback_pipeline
        run_fallback_pipeline()

    # Step 3 — Build vector store
    print("\n[5/5] Building RAG vector store...")
    from agent.retriever import build_vectorstore
    build_vectorstore()

    print("\n" + "=" * 50)
    print("  Pipeline complete. Run: streamlit run app.py")
    print("=" * 50)

if __name__ == "__main__":
    run()

