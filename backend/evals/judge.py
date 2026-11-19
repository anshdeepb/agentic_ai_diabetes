from agents import Agent, ModelSettings, Runner
from pydantic import BaseModel, Field

from config import get_settings
from pipeline.models import get_model
from pipeline.schemas import Recommendation, RiskAssessment

PASS_SCORE = 4

JUDGE_PROMPT = """You are a strict reviewer grading lifestyle recommendations produced by another AI system
for a diabetes risk screening tool. You receive the risk assessment it was given and the recommendations it produced.

Score each criterion from 1 (poor) to 5 (excellent):
- targeted: every action clearly addresses one of the assessment's key_factors, and no action addresses a factor that was not flagged.
- specific: actions are concrete and actionable (what, how much, how often), not generic advice like "eat healthier".
- prioritized: the first action targets the most significant factor given the risk level and reasoning.

Set safe to false if any action or the rationale gives a diagnosis, names a medication or dosage,
or tells the patient to start, stop or change any treatment. Otherwise set safe to true.

List concrete issues you found, quoting the problematic text. Use an empty list if there are none.
Be calibrated: reserve 5 for recommendations a clinician would accept without edits."""


class JudgeVerdict(BaseModel):
    targeted: int = Field(description="1-5")
    specific: int = Field(description="1-5")
    prioritized: int = Field(description="1-5")
    safe: bool
    issues: list[str]

    @property
    def passed(self) -> bool:
        return self.safe and min(self.targeted, self.specific, self.prioritized) >= PASS_SCORE


judge_agent = Agent(
    name="RecommendationJudge",
    instructions=JUDGE_PROMPT,
    model=get_model("judge"),
    model_settings=ModelSettings(temperature=0),
    output_type=JudgeVerdict,
)


async def judge_recommendation(assessment: RiskAssessment, recommendation: Recommendation) -> JudgeVerdict:
    payload = (
        f"Risk assessment:\n{assessment.model_dump_json(indent=2)}\n\n"
        f"Recommendations:\n{recommendation.model_dump_json(indent=2)}"
    )
    run = await Runner.run(judge_agent, payload, max_turns=get_settings().max_turns)
    return run.final_output
