"""
Ingestion boundary validation for ClinicalMind patient records.

Validates dicts as they cross from the extraction / quality pipeline into
``patients_summary.json`` (and any consumer that loads the same shape). This
layer catches schema drift, type mistakes, and inconsistent quality labels
before downstream agents or analytics run — the same class of guardrail
Arya Health implemented with Pydantic AI, reportedly cutting hallucination
rates from about 30% to under 1% by rejecting bad structured outputs early.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


class PatientRecord(BaseModel):
    """One patient row as produced for ``patients_summary.json``."""

    patient_id: str
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    length_of_stay: Optional[int] = Field(None, ge=0)
    prior_admissions: Optional[int] = Field(None, ge=0)
    has_follow_up: bool = False
    lives_alone: bool = False
    non_compliant: bool = False
    medications_raw: Optional[str] = None
    completeness_score: float = Field(..., ge=0.0, le=100.0)
    confidence_label: Literal["HIGH", "MEDIUM", "LOW"]
    quality_warning: str

    @model_validator(mode="after")
    def confidence_matches_score(self) -> PatientRecord:
        if self.confidence_label == "HIGH" and self.completeness_score < 80:
            raise ValueError(
                "confidence_label HIGH requires completeness_score >= 80"
            )
        if self.confidence_label == "MEDIUM" and self.completeness_score < 50:
            raise ValueError(
                "confidence_label MEDIUM requires completeness_score >= 50"
            )
        return self


class ValidatedPatientBatch(BaseModel):
    """Validated patient list with aggregate stats and parse failures."""

    records: List[PatientRecord]
    total: int
    high_confidence: int
    medium_confidence: int
    low_dropped: int
    validation_errors: List[str]

    @classmethod
    def from_raw_list(cls, raw_list: List[dict]) -> ValidatedPatientBatch:
        records: List[PatientRecord] = []
        validation_errors: List[str] = []

        for raw in raw_list:
            pid = raw.get("patient_id")
            pid_str = pid if isinstance(pid, str) else (str(pid) if pid is not None else "unknown")
            try:
                records.append(PatientRecord.model_validate(raw))
            except ValidationError as e:
                parts: list[str] = []
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else "record"
                    parts.append(f"{loc}: {err['msg']}")
                validation_errors.append(f"{pid_str}: {'; '.join(parts)}")

        high_confidence = sum(1 for r in records if r.confidence_label == "HIGH")
        medium_confidence = sum(1 for r in records if r.confidence_label == "MEDIUM")
        low_dropped = sum(1 for r in records if r.confidence_label == "LOW")

        return cls(
            records=records,
            total=len(records),
            high_confidence=high_confidence,
            medium_confidence=medium_confidence,
            low_dropped=low_dropped,
            validation_errors=validation_errors,
        )
