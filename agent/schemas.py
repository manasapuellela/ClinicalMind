"""
Structured output contracts for ClinicalMind LangGraph nodes.

These Pydantic models define the shape of agent responses so analysis is not an
opaque string: parsing and validation produce deterministic, inspectable objects
suited for UI rendering and audit trails. This mirrors the architectural pattern
Arya uses with Pydantic AI—structured outputs make scheduling (and here, clinical
analysis) agents more deterministic and auditable than free-form LLM text alone.
"""

from __future__ import annotations

import re
from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


class RiskFactor(BaseModel):
    """One identified clinical or utilization risk driver."""

    factor: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    detail: str


class PatientRiskAssessment(BaseModel):
    """Structured risk output for a single patient."""

    patient_id: str
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    risk_score: int = Field(ge=0, le=20)
    risk_factors: List[RiskFactor]
    data_quality_note: str
    recommended_intervention: str
    reasoning_summary: str

    @model_validator(mode="after")
    def data_quality_note_mentions_confidence_label(self) -> PatientRiskAssessment:
        note = self.data_quality_note
        if "confidence" not in note.lower():
            raise ValueError(
                "data_quality_note must mention confidence (e.g. tying the note to "
                "confidence_label from the patient record)."
            )
        if not re.search(r"\b(high|medium|low)\b", note, re.IGNORECASE):
            raise ValueError(
                "data_quality_note must state the record confidence level "
                "(HIGH, MEDIUM, or LOW)."
            )
        return self


class ClinicalAnalysisResponse(BaseModel):
    """Top-level structured response from analyze_node / followup_node."""

    query_understood: str
    assessments: List[PatientRiskAssessment]
    summary: str
    data_quality_warnings: List[str]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]

    def to_display_text(self) -> str:
        """Format as markdown for Streamlit chat."""
        lines: List[str] = [self.summary.strip(), ""]

        if self.assessments:
            lines.append(
                "| patient_id | risk_level | top risk factor | intervention |"
            )
            lines.append("| --- | --- | --- | --- |")
            for a in self.assessments:
                top_factor = a.risk_factors[0].factor if a.risk_factors else "—"
                intervention = a.recommended_intervention.replace(
                    "|", "\\|"
                )
                lines.append(
                    f"| {a.patient_id} | {a.risk_level} | "
                    f"{top_factor.replace('|', '\\|')} | {intervention} |"
                )
            lines.append("")

        if self.data_quality_warnings:
            lines.append("**Data quality warnings**")
            for w in self.data_quality_warnings:
                lines.append(f"- {w.strip()}")
            lines.append("")

        lines.append(f"**Overall confidence:** {self.confidence}")
        return "\n".join(lines).rstrip()
