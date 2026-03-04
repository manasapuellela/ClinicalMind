# 🏥 ClinicalMind — Patient Risk Intelligence Agent

AI-powered readmission risk scoring for hospital patients.
Built with PySpark, Delta Lake, LangGraph, RAG, and Claude.

## What It Does
Processes unstructured discharge summaries through a data pipeline (PySpark locally or Python-only fallback), scores every patient's 30-day readmission risk using an AI agent, and surfaces results through a conversational Streamlit interface.

## Tech Stack
| Layer | Technology |
|-------|------------|
| Data ingestion (local) | PySpark or Python-only fallback |
| Storage (local) | Delta Lake + JSON summary |
| Data quality | PySpark or fallback pipeline |
| Vector store | FAISS + sentence-transformers |
| Agent | LangGraph StateGraph |
| LLM | Claude (claude-sonnet-4-5) |
| Interface | Streamlit |

**Cloud deployment:** The app runs **end-to-end without PySpark**. It uses `pipeline/patient_loader.py` (Spark-free) to load patient data and `pipeline/bootstrap.py` to generate synthetic data, run the fallback pipeline, and build the RAG index on first run.

## How LangGraph Manages State
The agent uses a StateGraph with 4 nodes:
- **load_data** — loads processed patient JSON into shared state (via `patient_loader`)
- **retrieve** — RAG retrieval of clinical guidelines
- **analyze** — first-turn reasoning over all patients
- **followup** — subsequent turns with full memory

Conditional edges route between nodes based on conversation turn. State persists messages, patient data, and context across all turns.

## Project Layout (relevant to Cloud vs local)
- **`pipeline/patient_loader.py`** — Loads `patients_summary.json` (no PySpark). Used by the app and agent on Cloud and locally.
- **`pipeline/delta_writer.py`** — Writes Delta + JSON from Spark (local pipeline only; not imported on Cloud).
- **`pipeline/bootstrap.py`** — First-run setup: generates data, runs fallback pipeline, builds RAG. Used by the app when no data exists (e.g. fresh Streamlit Cloud deploy).
- **`pipeline/fallback_pipeline.py`** — Pure Python pipeline (no Spark). Used when Spark fails locally or by bootstrap on Cloud.

## Setup (local)

1. Clone the repo.
2. `pip install -r requirements.txt`  
   For **local** Spark pipeline you need PySpark and Java; install separately if you use `run_pipeline.py` with Spark.
3. Copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY`.
4. **(Optional)** Generate data and run the pipeline:
   - `python generate_data.py` — creates synthetic discharge summaries in `data/raw`
   - `python run_pipeline.py` — runs pipeline (Spark if available, else fallback) and builds RAG
5. Start the app: `python -m streamlit run app.py`  
   If you skip step 4, the app will **bootstrap** on first run (generate data, run fallback pipeline, build RAG).

## Deployment (Streamlit Cloud)

1. Push your code to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create a new app from your repo (`main`, main file `app.py`).
3. In **Secrets**, add: `ANTHROPIC_API_KEY=your_key`
4. Deploy. The app is **end-to-end**: on first load it generates synthetic data, runs the fallback pipeline, and builds the RAG index (no Spark, no Java).

## Disclaimer
Uses synthetic patient data only. Not for real clinical use.
