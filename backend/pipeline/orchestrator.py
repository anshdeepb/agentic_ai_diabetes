from agents import Agent, ModelSettings, RunContextWrapper, Runner

from config import get_settings
from memory.session import get_session
from memory.state import CachedAssessment, ConversationState, StateStore
from pipeline.models import get_model
from pipeline.recommendation import recommendation_agent
from pipeline.risk_assessment import RISK_PROMPT_VERSION, risk_agent
from pipeline.schemas import (
    ClarificationResult,
    Intent,
    OrchestratorDecision,
    PipelineResult,
    RecommendationResult,
    RiskAssessment,
    RiskAssessmentResult,
)
from pipeline.tools import patient_exists, record_hash

ORCHESTRATOR_PROMPT = """Classify a diabetes-related query into one intent:
- risk_only: the user wants to know how risky a patient's profile looks.
- recommendation: the user wants advice or next steps for a patient.

Extract the patient_id the user refers to. Patient ids are numbers written as strings, e.g. "0", "42".
- A newly named patient ("how about patient 5?") always wins over the active patient.
- If the user refers to a patient indirectly ("her", "them", "the same patient"), use the active patient.
- A short follow-up that only changes the patient keeps the intent of the previous user question.
- If no patient is mentioned and there is no active patient, set patient_id to JSON null, not the string "null".

Conversation state: {state}"""


def orchestrator_instructions(
    ctx: RunContextWrapper[ConversationState], agent: Agent[ConversationState]
) -> str:
    return ORCHESTRATOR_PROMPT.format(state=ctx.context.summary())


orchestrator_agent = Agent[ConversationState](
    name="Orchestrator",
    instructions=orchestrator_instructions,
    model=get_model("orchestrator"),
    model_settings=ModelSettings(temperature=0),
    output_type=OrchestratorDecision,
)


async def get_or_create_assessment(
    patient_id: str, state: ConversationState
) -> tuple[RiskAssessment, bool]:
    current_hash = record_hash(patient_id)
    model = get_model("risk")
    cached = state.assessments.get(patient_id)
    if cached and cached.matches(current_hash, RISK_PROMPT_VERSION, model):
        return cached.assessment, True

    run = await Runner.run(
        risk_agent,
        f"Assess risk for patient_id={patient_id}",
        max_turns=get_settings().max_turns,
    )
    assessment = run.final_output.model_copy(update={"patient_id": patient_id})
    state.assessments[patient_id] = CachedAssessment(
        assessment=assessment,
        record_hash=current_hash,
        prompt_version=RISK_PROMPT_VERSION,
        model=model,
    )
    return assessment, False


def summarize_result(result: PipelineResult) -> str:
    if isinstance(result, ClarificationResult):
        return result.message
    a = result.assessment
    text = f"Patient {a.patient_id}: {a.risk_level.value} risk ({', '.join(a.key_factors)})."
    if isinstance(result, RecommendationResult):
        text += f" Recommended: {'; '.join(result.recommendation.priority_actions)}"
    return text


async def orchestrate(query: str, session_id: str, store: StateStore) -> PipelineResult:
    settings = get_settings()
    state = await store.load(session_id)
    session = get_session(session_id)

    decision_run = await Runner.run(
        orchestrator_agent,
        query,
        context=state,
        session=session,
        max_turns=settings.max_turns,
    )
    decision: OrchestratorDecision = decision_run.final_output
    patient_id = decision.patient_id or state.active_patient_id

    result: PipelineResult
    if patient_id is None:
        result = ClarificationResult(
            message="Which patient would you like me to look at? For example: \"What is patient 0's risk?\""
        )
    elif not patient_exists(patient_id):
        result = ClarificationResult(message=f"I couldn't find a patient with id {patient_id}.")
    else:
        state.active_patient_id = patient_id
        assessment, from_cache = await get_or_create_assessment(patient_id, state)
        if decision.intent == Intent.risk_only:
            result = RiskAssessmentResult(assessment=assessment, from_cache=from_cache)
        else:
            rec_run = await Runner.run(
                recommendation_agent,
                assessment.model_dump_json(),
                max_turns=settings.max_turns,
            )
            recommendation = rec_run.final_output.model_copy(update={"patient_id": patient_id})
            result = RecommendationResult(
                assessment=assessment, recommendation=recommendation, from_cache=from_cache
            )

    await session.add_items([{"role": "assistant", "content": summarize_result(result)}])
    await store.save(state)
    return result
