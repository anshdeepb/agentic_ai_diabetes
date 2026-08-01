from agents import Agent, Runner
from pipeline.schemas import OrchestratorDecision, Intent
from pipeline.risk_assessment import risk_agent
from pipeline.recommendation import recommendation_agent

orchestration_prompt = """Classify a diabetes-related query into one of:
    - risk_only: user wants to know how risky their profile looks
    - recommendation_only: user wants advice and a risk assessment already exists
    - recommendation_needs_risk_first: user wants advice but no assessment exists yet
    Extract the patient_id mentioned in the query."""

orchestrator_agent = Agent(
    name="Orchestrator",
    instructions=orchestration_prompt,
    model="gpt-4o-mini",
    output_type=OrchestratorDecision
)

# orchestration logic

async def orchestrate(user_query: str):
    decision_result = await Runner.run(orchestrator_agent, user_query)
    decision = decision_result.final_output

    if decision.intent == Intent.risk_only:
        risk_result = await Runner.run(risk_agent, f"Assess risk for patient_id={decision.patient_id}")
        return risk_result.final_output

    risk_result = await Runner.run(risk_agent, f"Assess risk for patient_id={decision.patient_id}")
    assessment = risk_result.final_output

    recommendation_result = await Runner.run(recommendation_agent, assessment.model_dump_json())
    return recommendation_result.final_output

