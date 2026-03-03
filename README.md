# 🏥 ClinicalMind — Patient Risk Intelligence Agent

AI-powered readmission risk scoring for hospital patients.
Built with PySpark, Delta Lake, LangGraph, RAG, and Claude.

## What It Does
Processes unstructured discharge summaries through a PySpark pipeline,
scores every patient's 30-day readmission risk using an AI agent,
and surfaces results through a conversational Streamlit interface.

## Tech Stack
| Layer | Technology |
|---|---|
| Data Ingestion | PySpark |
| Storage | Delta Lake |
| Data Quality | PySpark quality scoring |
| Vector Store | FAISS + sentence-transformers |
| Agent | LangGraph StateGraph |
| LLM | Claude claude-sonnet-4-5 |
| Interface | Streamlit |

## How LangGraph Manages State
The agent uses a StateGraph with 4 nodes:
- **load_data** — loads processed patient JSON into shared state
- **retrieve** — RAG retrieval of clinical guidelines
- **analyze** — first-turn reasoning over all patients
- **followup** — subsequent turns with full memory

Conditional edges route between nodes based on conversation turn.
State persists messages, patient data, and context across all turns.

## Setup
1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
4. `python generate_data.py`
5. `python run_pipeline.py`
6. `streamlit run app.py`

## Deployment
Deploy free on Streamlit Cloud.
Add `ANTHROPIC_API_KEY` in Settings → Secrets.

## Disclaimer
Uses synthetic patient data only. Not for real clinical use.

