"""
LANGGRAPH AGENT
Defines the full StateGraph with nodes and routing logic.

NODES:
  load_data   — loads processed patient JSON into state
  retrieve    — RAG retrieval of clinical guidelines context  
  analyze     — Claude reasons over patients + context
  followup    — handles follow-up questions with full memory

FLOW:
  First message:  load_data → retrieve → analyze → END
  Follow-up:      retrieve → followup → END
"""

import json
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from agent.prompts import SYSTEM_PROMPT
from agent.retriever import retrieve_context
from agent.schemas import ClinicalAnalysisResponse
from agent.state import PAState
from pipeline.patient_loader import load_patients_validated, validate_and_load

logger = logging.getLogger(__name__)

load_dotenv()

# ── Model ──────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2048,
    temperature=0.1,
)

STRUCTURED_JSON_INSTRUCTION = """
STRUCTURED RESPONSE FORMAT (required):
Respond with valid JSON only — no markdown fences, no preamble, no text before or after the JSON object.

The JSON must have exactly this structure. For each field shown as HIGH|MEDIUM|LOW, use a single string: "HIGH", "MEDIUM", or "LOW". For risk_score use an integer from 0 to 20 (inclusive).

{
  "query_understood": "...",
  "assessments": [
    {
      "patient_id": "...",
      "risk_level": "HIGH|MEDIUM|LOW",
      "risk_score": 0-20,
      "risk_factors": [{"factor": "...", "severity": "HIGH|MEDIUM|LOW", "detail": "..."}],
      "data_quality_note": "...",
      "recommended_intervention": "...",
      "reasoning_summary": "..."
    }
  ],
  "summary": "...",
  "data_quality_warnings": ["..."],
  "confidence": "HIGH|MEDIUM|LOW"
}

assessments may be an empty array if the clinician's question is not patient-specific.
"""


def _llm_content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                txt = block.get("text")
                if txt is not None:
                    parts.append(str(txt))
        if parts:
            return "".join(parts)
    return str(content)


def clean_json(text: str) -> str:
    """
    Normalize model output so json.loads succeeds when fences or preamble/suffix exist.
    Order: strip → trim ```json / ``` fences → keep outermost {...} only.
    """
    s = text.strip()
    if s.lower().startswith("```json"):
        s = s[7:].lstrip("\n\r ")
    elif s.startswith("```"):
        s = s[3:].lstrip("\n\r ")
    s = s.rstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    first = s.find("{")
    if first != -1:
        s = s[first:]
    last = s.rfind("}")
    if last != -1:
        s = s[: last + 1]
    return s


# ── NODE 1: Load Data ──────────────────────────────────────────────────────
def load_data_node(state: PAState) -> PAState:
    """
    Loads processed patient records from the JSON summary.
    Injects them into state so all subsequent nodes can access them.
    Only runs on the first turn — data stays in state after that.
    """
    if state.get("patients"):
        return state  # Already loaded — skip

    try:
        patients = validate_and_load()
        batch = load_patients_validated()
        print(
            f"[ClinicalMind] Validation: {batch.total} passed, "
            f"{len(batch.validation_errors)} errors"
        )
        if batch.validation_errors:
            for err in batch.validation_errors:
                print(f"  [VALIDATION ERROR] {err}")
        print(f"Loaded {len(patients)} patient records into agent state.")
    except FileNotFoundError as e:
        patients = []
        print(f"Warning: {e}")

    return {
        **state,
        "patients": patients,
    }


# ── NODE 2: Retrieve ───────────────────────────────────────────────────────
def retrieve_node(state: PAState) -> PAState:
    """
    RAG retrieval node.
    Takes the current query and retrieves relevant clinical guidelines.
    This context is passed to the analyze/followup nodes.
    """
    query = state.get("current_query", "")
    if not query:
        return {**state, "context": ""}

    context = retrieve_context(query, k=3)
    return {
        **state,
        "context": context,
    }


# ── NODE 3: Analyze ────────────────────────────────────────────────────────
def analyze_node(state: PAState) -> PAState:
    """
    Core reasoning node for the first turn.
    Sends patient data + retrieved context + system prompt to Claude.
    Claude scores risk, explains reasoning, recommends interventions.
    Adds response to message history.
    """
    messages = state.get("messages", [])
    patients = state.get("patients", [])
    context = state.get("context", "")
    query = state.get("current_query", "")

    # Build patient data summary for the prompt
    # Limit to 20 patients to stay within context window
    patient_summary = json.dumps(patients[:20], indent=2, default=str)

    # Full prompt to Claude
    analysis_prompt = f"""
PATIENT DATA ({len(patients)} total records, showing first 20):
{patient_summary}

CLINICAL GUIDELINES CONTEXT:
{context}

CLINICIAN QUESTION:
{query}

Answer the clinician's question using the patient data above.
Apply the risk scoring logic from your system prompt.
Always surface data quality warnings for incomplete records.
{STRUCTURED_JSON_INSTRUCTION}
"""

    claude_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages + [
        HumanMessage(content=analysis_prompt)
    ]

    try:
        response = llm.invoke(claude_messages)
        raw = _llm_content_to_str(response.content)
        cleaned = clean_json(raw)
        try:
            parsed = json.loads(cleaned)
            response_text = ClinicalAnalysisResponse(**parsed).to_display_text()
        except (ValidationError, json.JSONDecodeError):
            logger.warning("Structured output validation failed, using raw response")
            response_text = raw
    except Exception as e:
        response_text = f"⚠️ Error connecting to Claude API: {str(e)}. Please check your API key."

    updated_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=response_text),
    ]

    return {
        **state,
        "messages": updated_messages,
        "response": response_text,
    }


# ── NODE 4: Follow-up ──────────────────────────────────────────────────────
def followup_node(state: PAState) -> PAState:
    """
    Follow-up node for subsequent questions.
    Has full conversation history so context is preserved naturally.
    Clinician can ask: 'tell me more about patient 3',
    'what intervention should we do for the CHF patients?', etc.
    """
    messages = state.get("messages", [])
    patients = state.get("patients", [])
    context = state.get("context", "")
    query = state.get("current_query", "")

    # For follow-ups we include the full patient data again
    # so Claude can reference any patient, not just ones mentioned before
    patient_summary = json.dumps(patients[:20], indent=2, default=str)

    followup_prompt = f"""
FULL PATIENT DATA (for reference):
{patient_summary}

ADDITIONAL CLINICAL CONTEXT:
{context}

FOLLOW-UP QUESTION:
{query}

Answer using the full conversation history above for context.
Reference specific patient IDs and data points in your answer.
{STRUCTURED_JSON_INSTRUCTION}
"""

    claude_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages + [
        HumanMessage(content=followup_prompt)
    ]

    try:
        response = llm.invoke(claude_messages)
        raw = _llm_content_to_str(response.content)
        cleaned = clean_json(raw)
        try:
            parsed = json.loads(cleaned)
            response_text = ClinicalAnalysisResponse(**parsed).to_display_text()
        except (ValidationError, json.JSONDecodeError):
            logger.warning("Structured output validation failed, using raw response")
            response_text = raw
    except Exception as e:
        response_text = f"⚠️ Error: {str(e)}"

    updated_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=response_text),
    ]

    return {
        **state,
        "messages": updated_messages,
        "response": response_text,
    }


# ── ROUTING LOGIC ──────────────────────────────────────────────────────────
def route_entry(state: PAState) -> Literal["load_data", "retrieve"]:
    """
    Conditional entry point.
    First message → load data first, then retrieve + analyze.
    Follow-up → skip data loading, go straight to retrieve + followup.
    """
    messages = state.get("messages", [])
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    if len(human_messages) == 0:
        return "load_data"
    else:
        return "retrieve"


def route_after_retrieve(state: PAState) -> Literal["analyze", "followup"]:
    """
    After retrieval, route to analyze (first turn) or followup (subsequent turns).
    """
    messages = state.get("messages", [])
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    if len(human_messages) == 0:
        return "analyze"
    else:
        return "followup"


# ── BUILD GRAPH ────────────────────────────────────────────────────────────
def build_graph():
    """
    Assembles the full LangGraph StateGraph.

    Flow diagram:
    START
      ↓ (conditional)
    load_data → retrieve → analyze → END
                    ↓
                followup → END
    """
    graph = StateGraph(PAState)

    # Register nodes
    graph.add_node("load_data", load_data_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("followup", followup_node)

    # Entry point: first turn loads data, follow-ups skip to retrieve
    graph.set_conditional_entry_point(
        route_entry,
        {
            "load_data": "load_data",
            "retrieve": "retrieve",
        }
    )

    # After loading data, always retrieve context
    graph.add_edge("load_data", "retrieve")

    # After retrieval, route to analyze or followup
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "analyze": "analyze",
            "followup": "followup",
        }
    )

    # Both analyze and followup end the turn
    graph.add_edge("analyze", END)
    graph.add_edge("followup", END)

    return graph.compile()


# Compiled graph — imported by app.py
clinical_graph = build_graph()
