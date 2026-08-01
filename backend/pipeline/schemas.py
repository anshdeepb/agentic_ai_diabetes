import os

from pydantic import BaseModel, Field 
from enum import Enum

class RiskLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"

class RiskAssessment(BaseModel):
    patient_id : str
    risk_level : RiskLevel
    key_factors : list[str] = Field(description="Which attributes are driving the risk level")
    reasoning : str
    used_cohort_context : bool = Field(description="Whether cohort comparison was needed")

class Recommendation(BaseModel):
    patient_id: str
    priority_actions: list[str] = Field(description="2-4 concrete, prioritized actions")
    rationale: str

class Intent(str, Enum):
    risk_only = "risk_only"
    recommendation_only = "recommendation_only"
    recommendation_needs_risk_first = "recommendation_needs_risk_first"

class OrchestratorDecision(BaseModel):
    intent: Intent
    patient_id: str
    reasoning: str