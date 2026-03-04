"""
E2E BOOTSTRAP — Cloud / first-run setup without Spark.
If patients_summary.json is missing, generates synthetic data,
runs the fallback pipeline, and builds the RAG vectorstore.
Makes Streamlit Cloud deployment work end-to-end without pre-committed data.
"""

import os

JSON_PATH = "data/processed/patients_summary.json"
RAW_DIR = "data/raw"


def ensure_data_ready():
    """
    Ensure patient JSON and RAG vectorstore exist. If not, generate data,
    run fallback pipeline, and build vectorstore. Idempotent — safe to call every run.
    """
    if os.path.exists(JSON_PATH):
        return

    # 1. Ensure raw data exists (generate if empty)
    if not os.path.isdir(RAW_DIR) or not [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]:
        import generate_data
        generate_data.run(50)

    # 2. Run fallback pipeline → writes patients_summary.json
    from pipeline.fallback_pipeline import run_fallback_pipeline
    run_fallback_pipeline()

    # 3. Build RAG vectorstore from clinical guidelines
    from agent.retriever import build_vectorstore
    build_vectorstore()
