# 🏥 ClinicalMind: Patient Risk Intelligence Agent

AI-powered readmission risk scoring for hospital patients.
Built with PySpark, Delta Lake, LangGraph, RAG, Claude, and Pydantic AI.

## What It Does
Processes unstructured discharge summaries through a multi-layer data pipeline (PySpark locally or Python-only fallback), validates every record at the ingestion boundary using Pydantic v2, scores each patient's 30-day readmission risk using a LangGraph agent backed by Claude, and enforces structured output contracts on every AI response, surfacing results through a conversational Streamlit interface.

## Architecture Overview

```
data/raw/*.txt
      ↓
Ingestion (PySpark or Python fallback)
      ↓
Extraction (regex → structured fields)
      ↓
Data Quality Scoring (completeness_score, confidence_label, is_scoreable)
      ↓  [LOW confidence records dropped]
Delta Lake (local Spark only) + patients_summary.json
      ↓
Pydantic Validation Boundary (PatientRecord, ValidatedPatientBatch)
      ↓
LangGraph State (patients: list[dict])
      ↓
RAG Retrieval (FAISS, k=3, clinical guidelines)
      ↓
Claude Prompt (20 patients + context + query)
      ↓
clean_json() → json.loads() → ClinicalAnalysisResponse validation
      ↓  [fallback to raw string on parse failure]
.to_display_text() → Streamlit chat bubble
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion (local) | PySpark or Python-only fallback |
| Storage (local) | Delta Lake + JSON summary |
| Data quality | Weighted completeness scoring |
| Ingestion validation | Pydantic v2 : PatientRecord, ValidatedPatientBatch |
| Vector store | FAISS + sentence-transformers |
| Agent orchestration | LangGraph StateGraph |
| LLM | Claude (claude-sonnet-4-5) via Anthropic API |
| Output validation | Pydantic v2 : ClinicalAnalysisResponse |
| Interface | Streamlit |

**Cloud deployment:** Runs end-to-end without PySpark. Uses `pipeline/patient_loader.py` (Spark-free) and `pipeline/bootstrap.py` to generate synthetic data, run the fallback pipeline, and build the RAG index on first run.

## Why Pydantic AI

After reading Arya Health's engineering blog ("From RAGs to Riches"), which documented how structured Pydantic output contracts dropped their agent hallucination rate from ~30% to under 1%, ClinicalMind added the same two-layer validation pattern:

**Layer 1 — Ingestion boundary** (`pipeline/schemas.py`):
Every patient dict is validated as a `PatientRecord` before it reaches the agent. Type enforcement, range checks (age 0-120, scores 0-100), and cross-field consistency (confidence_label must match completeness_score) run on every record. Invalid rows are logged and dropped : they never silently reach the LLM.

**Layer 2 — Agent output** (`agent/schemas.py`):
Claude is instructed to respond with structured JSON only. Every response is parsed through `clean_json()` (strips markdown fences, preamble, trailing text) then validated as a `ClinicalAnalysisResponse`. This enforces typed risk levels, per-patient assessments, and data quality notes on every turn. On validation failure the pipeline falls back to raw string and logs a warning, it never crashes silently.

## How LangGraph Manages State

The agent uses a StateGraph with 4 nodes:

| Node | Role |
|---|---|
| load_data | Loads and validates patient JSON into shared state via validate_and_load() |
| retrieve | RAG retrieval of clinical guidelines (FAISS, k=3) |
| analyze | First-turn reasoning — patients + context + query → Claude → structured output |
| followup | Subsequent turns with full conversation history |

Routing: first message → load_data → retrieve → analyze. Follow-up messages → retrieve → followup.

## Project Layout

```
agent/
  graph.py          — LangGraph StateGraph, nodes, routing, clean_json()
  schemas.py        — Pydantic output contracts (ClinicalAnalysisResponse)
  state.py          — PAState TypedDict
  prompts.py        — SYSTEM_PROMPT and structured JSON instruction
  retriever.py      — FAISS vectorstore build and retrieval

pipeline/
  schemas.py        — Pydantic ingestion contracts (PatientRecord, ValidatedPatientBatch)
  patient_loader.py — load_patients_json(), validate_and_load(), load_patients_validated()
  fallback_pipeline.py — Pure Python pipeline (no Spark)
  bootstrap.py      — First-run setup for Streamlit Cloud
  ingestion.py      — PySpark ingestion (local only)
  extractor.py      — PySpark field extraction (local only)
  quality_check.py  — PySpark quality scoring (local only)
  delta_writer.py   — Delta Lake write (local only)

data/
  raw/              — Synthetic discharge summaries (.txt) — gitignored
  processed/        — patients_summary.json, Delta Lake, vectorstore — gitignored
  knowledge/        — clinical_guidelines.txt (RAG source)

app.py              — Streamlit chat interface
generate_data.py    — Synthetic data generator (50 patients)
run_pipeline.py     — One-command local pipeline runner
```

## Local Setup

### Prerequisites
- Python 3.11+
- An Anthropic API key

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/manasapuellela/ClinicalMind.git
cd ClinicalMind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here

# 4. Generate synthetic data and run pipeline
python generate_data.py
python run_pipeline.py

# 5. Start the app
streamlit run app.py
```

The app will open at http://localhost:8501

**Skip step 4 if you want:** the app auto-bootstraps on first run — it generates data, runs the fallback pipeline, and builds the RAG index automatically.

### Optional: Test the Pydantic layer directly

```bash
# Test ingestion validation
python -c "
from pipeline.patient_loader import load_patients_validated
batch = load_patients_validated()
print(f'Total: {batch.total}, HIGH: {batch.high_confidence}, Errors: {len(batch.validation_errors)}')
"

# Test agent output schema
python -c "
from agent.schemas import ClinicalAnalysisResponse
print('agent/schemas.py loaded successfully')
"
```

## Cloud Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Go to share.streamlit.io, create a new app from your repo (main branch, app.py)
3. In **Secrets**, add: `ANTHROPIC_API_KEY=your_key`
4. Deploy — the app bootstraps end-to-end on first load

## Known Limitations

- **Regex extraction only** — field extraction uses pattern matching on the synthetic template. Real clinical notes would need NER or LLM-based extraction.
- **20-patient context window** — analyze and followup nodes embed the first 20 patients per prompt. All records exist in state but only 20 are visible to the model per turn.
- **Structured output is best-effort** — if Claude's JSON fails clean_json() + Pydantic validation, the pipeline falls back to raw string and logs a warning. No retry loop.
- **Synthetic data only** — not for real clinical use.

## Disclaimer
Uses synthetic patient data only. Not for real clinical use.
