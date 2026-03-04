"""
CLINICALMIND — STREAMLIT CHAT INTERFACE
The front-end for the Patient Risk Intelligence Agent.
Clinicians interact via natural language.
The LangGraph agent responds with risk assessments grounded in real data.
"""

import streamlit as st
import json
import os
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import clinical_graph
from pipeline.patient_loader import load_patients_json

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalMind — AI Readmission Risk Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, .stApp {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0f1117;
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #141820;
        border-right: 1px solid #1e2533;
    }

    /* Stat cards */
    .stat-card {
        background: #1a2035;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8em;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stat-label {
        font-size: 0.72em;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #718096;
        margin-top: 2px;
    }
    .stat-high  { color: #fc8181; }
    .stat-med   { color: #f6ad55; }
    .stat-low   { color: #68d391; }
    .stat-total { color: #63b3ed; }

    /* Chat bubbles */
    .user-bubble {
        background: #1a365d;
        border: 1px solid #2a4a7f;
        color: #bee3f8;
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 10px 0 10px 15%;
        font-size: 0.93em;
        line-height: 1.6;
    }
    .agent-bubble {
        background: #1a202c;
        border: 1px solid #2d3748;
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 16px 16px 16px 4px;
        margin: 10px 15% 10px 0;
        font-size: 0.93em;
        line-height: 1.7;
        font-family: 'IBM Plex Sans', sans-serif;
        white-space: pre-wrap;
    }

    /* Risk badges */
    .badge-high   { background:#742a2a; color:#fc8181; padding:3px 10px; border-radius:20px; font-size:0.78em; font-weight:600; }
    .badge-medium { background:#744210; color:#f6ad55; padding:3px 10px; border-radius:20px; font-size:0.78em; font-weight:600; }
    .badge-low    { background:#1c4532; color:#68d391; padding:3px 10px; border-radius:20px; font-size:0.78em; font-weight:600; }

    /* Pipeline status */
    .pipeline-ok  { background:#1c4532; color:#68d391; padding:8px 14px; border-radius:8px; font-size:0.82em; margin-bottom:12px; }
    .pipeline-err { background:#742a2a; color:#fc8181; padding:8px 14px; border-radius:8px; font-size:0.82em; margin-bottom:12px; }

    /* Quick action buttons */
    .stButton > button {
        background: #1a2035;
        border: 1px solid #2d3748;
        color: #a0aec0;
        border-radius: 8px;
        font-size: 0.82em;
        padding: 6px 12px;
        width: 100%;
        text-align: left;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #1e2d3d;
        border-color: #4299e1;
        color: #bee3f8;
    }

    /* Input box */
    .stChatInput input {
        background: #1a202c !important;
        border: 1px solid #2d3748 !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "display_messages": [],
    "patients": [],
    "pipeline_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── E2E bootstrap (Cloud / first run) ──────────────────────────────────────
if not os.path.exists("data/processed/patients_summary.json"):
    with st.spinner("First-time setup: generating data, running pipeline, building RAG..."):
        from pipeline.bootstrap import ensure_data_ready
        ensure_data_ready()

# ── Load patient data ──────────────────────────────────────────────────────
@st.cache_data
def load_patients():
    try:
        return load_patients_json(), None
    except FileNotFoundError:
        return [], "No patient data found. Run `python run_pipeline.py` locally or refresh so the app can bootstrap."


patients, load_error = load_patients()
if patients and not st.session_state.pipeline_loaded:
    st.session_state.patients = patients
    st.session_state.pipeline_loaded = True


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ClinicalMind")
    st.markdown("*Patient Risk Intelligence Agent*")
    st.markdown("---")

    # Pipeline status
    if load_error:
        st.markdown(f'<div class="pipeline-err">⚠️ Pipeline not run yet</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="pipeline-ok">✅ {len(patients)} patients loaded</div>',
                    unsafe_allow_html=True)

    # Risk stats
    if patients:
        def score_risk(p):
            points = 0
            if (p.get("prior_admissions") or 0) >= 3: points += 3
            if not p.get("has_follow_up"): points += 3
            if p.get("lives_alone"): points += 2
            if p.get("non_compliant"): points += 2
            if (p.get("length_of_stay") or 0) > 7: points += 2
            if (p.get("age") or 0) > 75: points += 2
            high_dx = ["congestive heart failure","copd","sepsis","myocardial infarction"]
            if any(dx in (p.get("diagnosis") or "").lower() for dx in high_dx): points += 3
            if points >= 7: return "HIGH"
            if points >= 3: return "MEDIUM"
            return "LOW"

        risks = [score_risk(p) for p in patients]
        high = risks.count("HIGH")
        med  = risks.count("MEDIUM")
        low  = risks.count("LOW")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-number stat-high">{high}</div><div class="stat-label">High Risk</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-number stat-low">{low}</div><div class="stat-label">Low Risk</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-number stat-med">{med}</div><div class="stat-label">Medium Risk</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-number stat-total">{len(patients)}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick actions
    st.markdown("**⚡ Quick Questions**")
    quick_questions = [
        "🚨 Show top 5 high risk patients",
        "📋 List patients with no follow-up",
        "👴 Which patients over 75 are high risk?",
        "💊 Show non-compliant patients",
        "📊 Give me a full risk summary",
        "⚠️ Which records have data quality issues?",
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}"):
            st.session_state["prefill"] = q.split(" ", 1)[1]
            st.rerun()

    st.markdown("---")

    # Reset
    if st.button("🔄 New Session", use_container_width=True):
        for k in ["messages", "display_messages"]:
            st.session_state[k] = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75em; color:#4a5568; line-height:1.6'>
    PySpark/Delta (local) · LangGraph · Claude · FAISS RAG<br><br>
    ⚠️ Synthetic data only.<br>Not for clinical use.
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────
st.markdown("# Patient Risk Intelligence")

if load_error:
    st.error(f"⚠️ {load_error}")
    st.code("python run_pipeline.py   # or refresh the page to trigger first-time bootstrap", language="bash")
    st.stop()

# Welcome message
if not st.session_state.display_messages:
    st.markdown("""
    <div style='background:#1a2035; border:1px solid #2d3748; border-radius:12px; 
                padding:20px 24px; margin:16px 0; color:#a0aec0; font-size:0.9em; line-height:1.8'>
    👋 <strong style='color:#e2e8f0'>ClinicalMind is ready.</strong><br><br>
    Ask me anything about your patients:<br>
    • <em>"Who are the highest risk patients right now?"</em><br>
    • <em>"Why is PAT-0023 flagged as high risk?"</em><br>
    • <em>"Which CHF patients don't have follow-up appointments?"</em><br>
    • <em>"Show me all patients with data quality warnings"</em>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for role, content in st.session_state.display_messages:
    if role == "user":
        st.markdown(f'<div class="user-bubble">🧑‍⚕️ {content}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="agent-bubble">{content}</div>',
                    unsafe_allow_html=True)

# ── Input handling ─────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask about your patients...")
if prefill and not user_input:
    user_input = prefill

if user_input:
    st.session_state.display_messages.append(("user", user_input))

    with st.spinner("🔍 Analyzing patient data..."):
        try:
            current_state = {
                "messages":      st.session_state.messages,
                "patients":      st.session_state.patients,
                "context":       "",
                "current_query": user_input,
                "response":      "",
                "next_node":     "",
            }

            result = clinical_graph.invoke(current_state)

            st.session_state.messages = result["messages"]
            response_text = result.get("response", "No response generated.")
            st.session_state.display_messages.append(("agent", response_text))

        except Exception as e:
            err = f"⚠️ Error: {str(e)}\n\nCheck your ANTHROPIC_API_KEY in the .env file."
            st.session_state.display_messages.append(("agent", err))

    st.rerun()

