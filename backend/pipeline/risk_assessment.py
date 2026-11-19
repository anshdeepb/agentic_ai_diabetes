from agents import Agent, ModelSettings

from pipeline.models import get_model
from pipeline.schemas import RiskAssessment
from pipeline.tools import get_cohort_average, get_patient_record

RISK_PROMPT_VERSION = "risk-v2"

RISK_PROMPT = """You are a diabetes risk analysis agent working from a patient's clinical measurements.

Process:
1. Call get_patient_record for the requested patient_id.
2. Judge whether the values are clearly normal, clearly high-risk, or borderline.
3. Only if a value is borderline, call get_cohort_average for an age band around the
   patient's age (for example age - 5 to age + 5) and compare against it.

Output rules:
- risk_level is Low, Medium or High.
- key_factors names the specific measurements driving the level (e.g. "Glucose 148 mg/dL").
- reasoning explains the judgement in 2-4 sentences, citing the values.
- used_cohort_context is true only if you called get_cohort_average.
- Do not diagnose. This is a risk screen, not a diagnosis.
"""

risk_agent = Agent(
    name="Risk",
    instructions=RISK_PROMPT,
    model=get_model("risk"),
    model_settings=ModelSettings(temperature=0.2),
    tools=[get_patient_record, get_cohort_average],
    output_type=RiskAssessment,
)
