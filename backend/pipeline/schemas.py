import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class RiskAssessment(BaseModel):
    patient_id: str
    risk_level: RiskLevel
    key_factors: list[str] = Field(description="Which attributes are driving the risk level")
    reasoning: str
    used_cohort_context: bool = Field(description="Whether cohort comparison was needed")


class Recommendation(BaseModel):
    patient_id: str
    priority_actions: list[str] = Field(description="2-4 concrete, prioritized actions")
    rationale: str


class Intent(str, Enum):
    risk_only = "risk_only"
    recommendation = "recommendation"


class OrchestratorDecision(BaseModel):
    intent: Intent
    patient_id: str | None = Field(
        description="Patient the user is asking about, or null if the query does not say "
        "and there is no active patient to fall back on."
    )
    reasoning: str

    @field_validator("patient_id", mode="before")
    @classmethod
    def normalize_patient_id(cls, value: object) -> str | None:
        if value is None:
            return None
        digits = re.search(r"\d+", str(value))
        return digits.group(0) if digits else None


class RiskAssessmentResult(BaseModel):
    type: Literal["risk_assessment"] = "risk_assessment"
    assessment: RiskAssessment
    from_cache: bool = False


class RecommendationResult(BaseModel):
    type: Literal["recommendation"] = "recommendation"
    assessment: RiskAssessment
    recommendation: Recommendation
    from_cache: bool = Field(False, description="Whether the risk assessment was reused")


class ClarificationResult(BaseModel):
    type: Literal["clarification"] = "clarification"
    message: str


PipelineResult = Annotated[
    RiskAssessmentResult | RecommendationResult | ClarificationResult,
    Field(discriminator="type"),
]
